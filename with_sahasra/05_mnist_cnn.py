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

from shared.mnist_cnn import (
    CHECKPOINT_DIR,
    count_parameters,
    init_params,
    load_mnist_dataset,
    logits,
    loss_and_accuracy,
    summary_dict,
    train_step,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small CNN on real MNIST remotely with Sahasra.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--steps-per-execution", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_DIR / "mnist_cnn_sahasra.npz",
        help="Where to save the trained model checkpoint as a local .npz file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "shared" / "data" / "mnist",
        help="Directory where torchvision will cache the MNIST download.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_start = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "dataset_prepare",
                "data_dir": str(args.data_dir),
                "train_limit": args.train_limit,
                "val_limit": args.val_limit,
                "mode": "with_sahasra",
            }
        ),
        flush=True,
    )
    dataset = load_mnist_dataset(
        data_dir=args.data_dir,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )
    print(
        json.dumps(
            {
                "event": "dataset_ready",
                "train_examples": int(dataset["train_x"].shape[0]),
                "val_examples": int(dataset["val_x"].shape[0]),
                "backend": jax.default_backend(),
                "mode": "with_sahasra",
            }
        ),
        flush=True,
    )

    params = init_params(jax.random.PRNGKey(args.seed))
    learning_rate = np.asarray(args.learning_rate, dtype=np.float32)
    shuffle_rng = np.random.default_rng(args.seed + 41)
    logits_jit = jax.jit(logits)

    runtime = sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    )
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

    trainer = sahasra.trainer(
        train_step,
        eval_fn=loss_and_accuracy,
        runtime=runtime,
        chunk_arity=2,
        steps_per_execution=args.steps_per_execution,
    )
    val_x_ref, val_y_ref = runtime.pin_tensors([dataset["val_x"], dataset["val_y"]])

    state_refs = params
    final_params = None
    last_execution = None
    total_billed_inr = 0.0
    final_val_loss = None
    final_val_accuracy = None
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            order = shuffle_rng.permutation(dataset["train_x"].shape[0])
            shuffled_x = dataset["train_x"][order]
            shuffled_y = dataset["train_y"][order]
            epoch_result = trainer.run_epoch_and_evaluate(
                state_refs,
                shuffled_x,
                shuffled_y,
                batch_size=args.batch_size,
                static_args=(learning_rate,),
                eval_args=(val_x_ref, val_y_ref),
            )
            state_refs = epoch_result.state_tree
            last_execution = epoch_result.epoch_result.execution
            (train_loss, train_accuracy), (val_loss, val_accuracy) = epoch_result.materialize_metrics()
            billed_amount = float(last_execution.billed_amount_inr or 0.0)
            total_billed_inr += billed_amount
            final_val_loss = float(np.asarray(val_loss))
            final_val_accuracy = float(np.asarray(val_accuracy))
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "backend": jax.default_backend(),
                        "train_loss": float(np.asarray(train_loss)),
                        "train_accuracy": float(np.asarray(train_accuracy)),
                        "val_loss": final_val_loss,
                        "val_accuracy": final_val_accuracy,
                        "runtime_mode": last_execution.runtime_mode,
                        "billing_status": last_execution.billing_status,
                        "billed_amount_inr": billed_amount,
                        "chunk_calls": epoch_result.chunk_calls,
                        "steps_per_execution": epoch_result.steps_per_execution,
                        "epoch_elapsed_sec": time.perf_counter() - epoch_start,
                        "mode": "with_sahasra",
                    }
                ),
                flush=True,
            )

        final_params = epoch_result.epoch_result.materialize_state()
        checkpoint_path = sahasra.save_checkpoint(args.checkpoint_path, final_params, runtime=runtime)
        print(
            json.dumps(
                {
                    "event": "checkpoint_saved",
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_format": "npz_via_sahasra.save_checkpoint",
                    "parameter_count": count_parameters(final_params),
                    "mode": "with_sahasra",
                }
            ),
            flush=True,
        )
        sample_logits = logits_jit(final_params, dataset["val_x"][:8])
        sample_predictions = np.asarray(jnp.argmax(sample_logits, axis=-1), dtype=np.int32).tolist()
        print(
            json.dumps(
                {
                    "status": "completed",
                    "example": "05_mnist_cnn",
                    "mode": "with_sahasra",
                    "session_id": runtime.session.session.id,
                    "worker_id": runtime.session.session.worker_id,
                    "total_elapsed_sec": time.perf_counter() - total_start,
                    "total_billed_inr": round(total_billed_inr, 4),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_format": "npz_via_sahasra.save_checkpoint",
                    "parameter_count": count_parameters(final_params),
                    **summary_dict(dataset, final_params),
                    "final_runtime_mode": last_execution.runtime_mode if last_execution is not None else None,
                    "final_customer_summary": getattr(last_execution, "customer_summary", None),
                    "final_val_loss": final_val_loss,
                    "final_val_accuracy": final_val_accuracy,
                    "sample_targets": dataset["val_y"][:8].astype(int).tolist(),
                    "sample_predictions": sample_predictions,
                },
                indent=2,
            )
        )
    finally:
        close_response = runtime.close()
        if close_response is not None:
            print(
                json.dumps(
                    {
                        "close": {
                            "session_id": close_response.session_id,
                            "status": close_response.status,
                            "deleted_tensor_count": len(close_response.deleted_tensor_ids),
                        }
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
