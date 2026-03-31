from __future__ import annotations

import argparse
import json
import math
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.mnist_cnn import init_params, iterate_minibatches, load_mnist_dataset, logits, loss_and_accuracy, summary_dict, train_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small CNN on real MNIST locally with JAX.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--log-every-steps", type=int, default=5)
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--eval-subset", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--val-limit", type=int, default=1000)
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
                "mode": "without_sahasra",
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
                "mode": "without_sahasra",
            }
        ),
        flush=True,
    )
    params = init_params(jax.random.PRNGKey(args.seed))
    train_step_jit = jax.jit(train_step)
    eval_step_jit = jax.jit(loss_and_accuracy)
    logits_jit = jax.jit(logits)
    learning_rate = np.asarray(args.learning_rate, dtype=np.float32)
    rng = np.random.default_rng(args.seed + 29)
    steps_per_epoch = math.ceil(dataset["train_x"].shape[0] / args.batch_size)
    eval_subset = max(1, min(args.eval_subset, dataset["val_x"].shape[0]))
    final_val_loss = None
    final_val_accuracy = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        last_loss = None
        last_accuracy = None
        for step, (batch_x, batch_y) in enumerate(
            iterate_minibatches(
            dataset["train_x"],
            dataset["train_y"],
            batch_size=args.batch_size,
            rng=rng,
            ),
            start=1,
        ):
            params, loss, accuracy = train_step_jit(params, batch_x, batch_y, learning_rate)
            last_loss = float(np.asarray(loss))
            last_accuracy = float(np.asarray(accuracy))

            if args.log_every_steps > 0 and (step % args.log_every_steps == 0 or step == 1 or step == steps_per_epoch):
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch,
                            "epochs": args.epochs,
                            "step": step,
                            "steps_per_epoch": steps_per_epoch,
                            "backend": jax.default_backend(),
                            "train_loss": last_loss,
                            "train_accuracy": last_accuracy,
                            "elapsed_sec": time.perf_counter() - epoch_start,
                            "mode": "without_sahasra",
                        }
                    ),
                    flush=True,
                )

            if args.eval_every_steps > 0 and step < steps_per_epoch and step % args.eval_every_steps == 0:
                eval_loss, eval_accuracy = eval_step_jit(
                    params,
                    dataset["val_x"][:eval_subset],
                    dataset["val_y"][:eval_subset],
                )
                print(
                    json.dumps(
                        {
                            "event": "mid_epoch_eval",
                            "epoch": epoch,
                            "epochs": args.epochs,
                            "step": step,
                            "steps_per_epoch": steps_per_epoch,
                            "backend": jax.default_backend(),
                            "eval_subset": eval_subset,
                            "val_loss": float(np.asarray(eval_loss)),
                            "val_accuracy": float(np.asarray(eval_accuracy)),
                            "elapsed_sec": time.perf_counter() - epoch_start,
                            "mode": "without_sahasra",
                        }
                    ),
                    flush=True,
                )

        val_loss, val_accuracy = eval_step_jit(params, dataset["val_x"], dataset["val_y"])
        final_val_loss = float(np.asarray(val_loss))
        final_val_accuracy = float(np.asarray(val_accuracy))
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "backend": jax.default_backend(),
                    "last_batch_loss": last_loss,
                    "last_batch_accuracy": last_accuracy,
                    "val_loss": final_val_loss,
                    "val_accuracy": final_val_accuracy,
                    "epoch_elapsed_sec": time.perf_counter() - epoch_start,
                    "mode": "without_sahasra",
                }
            ),
            flush=True,
        )

    sample_logits = logits_jit(params, dataset["val_x"][:8])
    sample_predictions = np.asarray(jnp.argmax(sample_logits, axis=-1), dtype=np.int32).tolist()

    print(
        json.dumps(
            {
                "status": "completed",
                "example": "05_mnist_cnn",
                "mode": "without_sahasra",
                "backend": jax.default_backend(),
                "total_elapsed_sec": time.perf_counter() - total_start,
                "final_val_loss": final_val_loss,
                "final_val_accuracy": final_val_accuracy,
                **summary_dict(dataset, params),
                "sample_targets": dataset["val_y"][:8].astype(int).tolist(),
                "sample_predictions": sample_predictions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
