from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.getenv("SAHASRA_EXAMPLES_FORCE_CPU") == "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.mnist_cnn import CHECKPOINT_DIR, load_mnist_dataset, load_params, logits


def infer_logits(
    conv1_w,
    conv1_b,
    conv2_w,
    conv2_b,
    dense1_w,
    dense1_b,
    dense2_w,
    dense2_b,
    images,
):
    params = {
        "conv1": {"w": conv1_w, "b": conv1_b},
        "conv2": {"w": conv2_w, "b": conv2_b},
        "dense1": {"w": dense1_w, "b": dense1_b},
        "dense2": {"w": dense2_w, "b": dense2_b},
    }
    return logits(params, images)


def _numeric_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    array = np.asarray(value)
    if array.dtype == object and array.shape == ():
        array = np.asarray(array.item())
    return np.asarray(array, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run separate Sahasra inference from a saved MNIST CNN checkpoint.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_DIR / "mnist_cnn_sahasra.npz",
        help="Path to the saved .npz checkpoint produced by 05_mnist_cnn.py.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "shared" / "data" / "mnist",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_start = time.perf_counter()
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint_path}. Run with_sahasra/05_mnist_cnn.py first."
        )

    params = load_params(args.checkpoint_path)
    print(
        json.dumps(
            {
                "event": "checkpoint_loaded",
                "checkpoint_path": str(args.checkpoint_path),
                "mode": "with_sahasra",
            }
        ),
        flush=True,
    )

    dataset = load_mnist_dataset(data_dir=args.data_dir, train_limit=1, val_limit=args.val_limit)
    start = max(0, min(args.offset, dataset["val_x"].shape[0] - 1))
    end = min(start + args.batch_size, dataset["val_x"].shape[0])
    batch_x = dataset["val_x"][start:end]
    batch_y = dataset["val_y"][start:end]

    print(
        json.dumps(
            {
                "event": "inference_batch_ready",
                "batch_size": int(batch_x.shape[0]),
                "offset": start,
                "checkpoint_path": str(args.checkpoint_path),
                "backend": jax.default_backend(),
                "mode": "with_sahasra",
            }
        ),
        flush=True,
    )

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        print(
            json.dumps(
                {
                    "event": "runtime_connected",
                    "session_id": runtime.session.session.id,
                    "worker_id": runtime.session.session.worker_id,
                    "gpu_class": runtime.session.session.gpu_class,
                    "region": runtime.session.session.region,
                    "mode": "with_sahasra",
                }
            ),
            flush=True,
        )
        remote_infer = sahasra.jit(infer_logits, runtime=runtime, output_mode="remote")
        predict_start = time.perf_counter()
        inference_result = remote_infer.remote(
            params["conv1"]["w"],
            params["conv1"]["b"],
            params["conv2"]["w"],
            params["conv2"]["b"],
            params["dense1"]["w"],
            params["dense1"]["b"],
            params["dense2"]["w"],
            params["dense2"]["b"],
            batch_x,
        )
        scores = _numeric_array(inference_result.materialize())
        predictions = np.asarray(np.argmax(scores, axis=-1), dtype=np.int32)
        predict_elapsed = time.perf_counter() - predict_start
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        eval_loss = float(-np.mean(log_probs[np.arange(batch_y.shape[0]), batch_y]))
        eval_accuracy = float(np.mean(predictions == batch_y))
        total_billed_inr = float(inference_result.execution.billed_amount_inr or 0.0)

        print(
            json.dumps(
                {
                    "status": "completed",
                    "example": "05_mnist_cnn_inference",
                    "mode": "with_sahasra",
                    "checkpoint_path": str(args.checkpoint_path),
                    "batch_size": int(batch_x.shape[0]),
                    "offset": start,
                    "total_elapsed_sec": time.perf_counter() - total_start,
                    "predict_elapsed_sec": predict_elapsed,
                    "session_id": runtime.session.session.id,
                    "worker_id": runtime.session.session.worker_id,
                    "runtime_mode": inference_result.execution.runtime_mode,
                    "billing_status": inference_result.execution.billing_status,
                    "billed_amount_inr": round(total_billed_inr, 4),
                    "eval_loss": float(np.asarray(eval_loss)),
                    "eval_accuracy": float(np.asarray(eval_accuracy)),
                    "sample_targets": batch_y.astype(int).tolist(),
                    "sample_predictions": predictions.astype(int).tolist(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
