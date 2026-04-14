from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if os.getenv("SAHASRA_EXAMPLES_FORCE_CPU") == "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.scan_mlp import (
    accuracy,
    build_batch_indices,
    confusion_matrix,
    init_adam,
    init_params,
    make_gaussian_mixture_dataset,
    parameter_count,
    train_many,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote scan-based mini-batch MLP training with Sahasra.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-examples", type=int, default=2048)
    parser.add_argument("--val-examples", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "shared" / "checkpoints" / "scan_mlp_sahasra.npz",
    )
    parser.add_argument("--skip-runtime-status", action="store_true")
    return parser


def check_runtime_status(base_url: str) -> dict[str, object] | None:
    status_url = f"{base_url.rstrip('/')}/runtime-status"
    try:
        with urllib.request.urlopen(status_url, timeout=10) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(
            json.dumps(
                {
                    "event": "runtime_status_unavailable",
                    "status_url": status_url,
                    "error": str(exc),
                    "mode": "with_sahasra",
                }
            ),
            flush=True,
        )
        return None
    return json.loads(payload)


def main() -> None:
    args = build_parser().parse_args()
    total_start = time.perf_counter()
    base_url = os.getenv("SAHASRA_API_URL", "https://www.sahasra.dev")
    api_key = os.getenv("SAHASRA_API_BEARER_TOKEN")
    if not api_key:
        raise RuntimeError("SAHASRA_API_BEARER_TOKEN is required for remote Sahasra training.")

    if not args.skip_runtime_status:
        runtime_status = check_runtime_status(base_url)
        if runtime_status is not None:
            print(
                json.dumps(
                    {
                        "event": "runtime_status",
                        "status": runtime_status.get("status"),
                        "healthy_worker_count": runtime_status.get("healthy_worker_count"),
                        "primary_worker_status": runtime_status.get("primary_worker_status"),
                        "mode": "with_sahasra",
                    }
                ),
                flush=True,
            )
            if runtime_status.get("status") in {"reconnecting", "unavailable"}:
                raise RuntimeError(f"Sahasra runtime is not available: {runtime_status}")

    dataset = make_gaussian_mixture_dataset(
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    params = init_params(
        jax.random.PRNGKey(args.seed),
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
    )
    m, v, step = init_adam(params)
    batch_indices = build_batch_indices(
        train_examples=dataset["train_x"].shape[0],
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed + 17,
    )
    steps_per_epoch = dataset["train_x"].shape[0] // args.batch_size
    learning_rate = np.asarray(args.learning_rate, dtype=np.float32)

    print(
        json.dumps(
            {
                "event": "dataset_ready",
                "train_examples": int(dataset["train_x"].shape[0]),
                "val_examples": int(dataset["val_x"].shape[0]),
                "feature_dim": args.feature_dim,
                "num_classes": args.num_classes,
                "epochs": args.epochs,
                "steps_per_epoch": steps_per_epoch,
                "total_train_steps": int(batch_indices.shape[0]),
                "batch_index_payload_bytes": int(batch_indices.nbytes),
                "parameter_count": parameter_count(params),
                "backend": jax.default_backend(),
                "mode": "with_sahasra",
            }
        ),
        flush=True,
    )

    with sahasra.SahasraRuntime.connect(base_url=base_url, api_key=api_key, gpu_class="g5", region="ap-south-1") as runtime:
        print(
            json.dumps(
                {
                    "event": "runtime_connected",
                    "session_id": runtime.session.session.id,
                    "worker_id": runtime.session.session.worker_id,
                    "mode": "with_sahasra",
                }
            ),
            flush=True,
        )
        pin_start = time.perf_counter()
        params_ref = sahasra.device_put(params, runtime=runtime)
        m_ref = sahasra.device_put(m, runtime=runtime)
        v_ref = sahasra.device_put(v, runtime=runtime)
        train_x_ref = sahasra.device_put(dataset["train_x"], runtime=runtime)
        train_y_ref = sahasra.device_put(dataset["train_y"], runtime=runtime)
        val_x_ref = sahasra.device_put(dataset["val_x"], runtime=runtime)
        val_y_ref = sahasra.device_put(dataset["val_y"], runtime=runtime)
        pin_elapsed = time.perf_counter() - pin_start

        remote_train = runtime.jit(train_many)
        remote_eval = runtime.jit(accuracy)

        train_start = time.perf_counter()
        trained_params, _, _, _, losses = remote_train(
            params_ref,
            m_ref,
            v_ref,
            step,
            train_x_ref,
            train_y_ref,
            batch_indices,
            learning_rate,
        )
        train_elapsed = time.perf_counter() - train_start

        eval_start = time.perf_counter()
        val_accuracy = remote_eval(trained_params, val_x_ref, val_y_ref)
        eval_elapsed = time.perf_counter() - eval_start

        checkpoint_start = time.perf_counter()
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = sahasra.save_checkpoint(args.checkpoint, trained_params)
        checkpoint_elapsed = time.perf_counter() - checkpoint_start

    epoch_losses = np.asarray(losses).reshape(args.epochs, steps_per_epoch).mean(axis=1)
    for epoch, loss in enumerate(epoch_losses, start=1):
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "mean_train_loss": float(loss),
                    "mode": "with_sahasra",
                }
            ),
            flush=True,
        )

    loaded_params = sahasra.load_checkpoint(checkpoint_path)
    matrix = confusion_matrix(loaded_params, dataset["val_x"], dataset["val_y"], num_classes=args.num_classes)
    print(
        json.dumps(
            {
                "status": "completed",
                "example": "11_scan_mlp_training",
                "mode": "with_sahasra",
                "epochs": args.epochs,
                "steps_per_epoch": steps_per_epoch,
                "total_train_steps": int(batch_indices.shape[0]),
                "pin_elapsed_sec": pin_elapsed,
                "remote_train_elapsed_sec": train_elapsed,
                "remote_eval_elapsed_sec": eval_elapsed,
                "checkpoint_elapsed_sec": checkpoint_elapsed,
                "total_elapsed_sec": time.perf_counter() - total_start,
                "checkpoint_path": str(checkpoint_path),
                "final_train_loss": float(np.asarray(losses)[-1]),
                "val_accuracy": float(np.asarray(val_accuracy)),
                "confusion_matrix": matrix.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

