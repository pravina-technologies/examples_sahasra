from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.getenv("SAHASRA_EXAMPLES_FORCE_CPU") == "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

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
    parser = argparse.ArgumentParser(description="Local scan-based mini-batch MLP training without Sahasra.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-examples", type=int, default=2048)
    parser.add_argument("--val-examples", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_start = time.perf_counter()
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

    train_jit = jax.jit(train_many)
    eval_jit = jax.jit(accuracy)

    train_start = time.perf_counter()
    params, m, v, step, losses = train_jit(
        params,
        m,
        v,
        step,
        dataset["train_x"],
        dataset["train_y"],
        batch_indices,
        learning_rate,
    )
    val_accuracy = eval_jit(params, dataset["val_x"], dataset["val_y"])
    train_elapsed = time.perf_counter() - train_start

    epoch_losses = np.asarray(losses).reshape(args.epochs, steps_per_epoch).mean(axis=1)
    for epoch, loss in enumerate(epoch_losses, start=1):
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "backend": jax.default_backend(),
                    "mean_train_loss": float(loss),
                    "mode": "without_sahasra",
                }
            ),
            flush=True,
        )

    matrix = confusion_matrix(params, dataset["val_x"], dataset["val_y"], num_classes=args.num_classes)
    print(
        json.dumps(
            {
                "status": "completed",
                "example": "06_scan_mlp_training",
                "mode": "without_sahasra",
                "backend": jax.default_backend(),
                "epochs": args.epochs,
                "steps_per_epoch": steps_per_epoch,
                "total_train_steps": int(batch_indices.shape[0]),
                "train_elapsed_sec": train_elapsed,
                "total_elapsed_sec": time.perf_counter() - total_start,
                "train_examples": int(dataset["train_x"].shape[0]),
                "val_examples": int(dataset["val_x"].shape[0]),
                "parameter_count": parameter_count(params),
                "final_train_loss": float(np.asarray(losses)[-1]),
                "val_accuracy": float(np.asarray(val_accuracy)),
                "confusion_matrix": matrix.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

