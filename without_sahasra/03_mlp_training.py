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

from shared.synthetic_mlp import init_params, iterate_minibatches, loss_and_accuracy, make_dataset, summary_dict, train_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local MLP training example without Sahasra.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_start = time.perf_counter()
    dataset = make_dataset(seed=args.seed)
    params = init_params(jax.random.PRNGKey(args.seed), hidden_dim=32)
    train_step_jit = jax.jit(train_step)
    eval_step_jit = jax.jit(loss_and_accuracy)
    learning_rate = np.asarray(args.learning_rate, dtype=np.float32)
    rng = np.random.default_rng(args.seed + 13)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        last_loss = None
        last_accuracy = None
        for batch_x, batch_y in iterate_minibatches(
            dataset["train_x"],
            dataset["train_y"],
            batch_size=args.batch_size,
            rng=rng,
        ):
            params, loss, accuracy = train_step_jit(params, batch_x, batch_y, learning_rate)
            last_loss = float(np.asarray(loss))
            last_accuracy = float(np.asarray(accuracy))

        val_loss, val_accuracy = eval_step_jit(params, dataset["val_x"], dataset["val_y"])
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "backend": jax.default_backend(),
                    "last_batch_loss": last_loss,
                    "last_batch_accuracy": last_accuracy,
                    "val_loss": float(np.asarray(val_loss)),
                    "val_accuracy": float(np.asarray(val_accuracy)),
                    "epoch_elapsed_sec": time.perf_counter() - epoch_start,
                    "mode": "without_sahasra",
                }
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "status": "completed",
                "example": "03_mlp_training",
                "mode": "without_sahasra",
                "backend": jax.default_backend(),
                "total_elapsed_sec": time.perf_counter() - total_start,
                **summary_dict(dataset, params),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
