from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.tiny_transformer import (
    build_char_dataset,
    count_parameters,
    eval_step,
    init_params,
    iterate_minibatches,
    sample_text,
    train_step,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local tiny transformer example without Sahasra.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--sample-steps", type=int, default=80)
    parser.add_argument("--prompt", default="Sahasra ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parents[1] / "shared" / "data" / "corpus.txt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = build_char_dataset(corpus_path=args.corpus, context_length=args.context_length, stride=args.stride)
    params = init_params(
        jax.random.PRNGKey(args.seed),
        vocab_size=dataset["vocab_size"],
        context_length=args.context_length,
    )
    train_step_jit = jax.jit(lambda p, x, y, lr: train_step(p, x, y, lr))
    eval_step_jit = jax.jit(lambda p, x, y: eval_step(p, x, y))
    learning_rate = np.asarray(args.learning_rate, dtype=np.float32)
    rng = np.random.default_rng(args.seed + 17)

    for epoch in range(1, args.epochs + 1):
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
                    "last_batch_loss": last_loss,
                    "last_batch_accuracy": last_accuracy,
                    "val_loss": float(np.asarray(val_loss)),
                    "val_accuracy": float(np.asarray(val_accuracy)),
                    "mode": "without_sahasra",
                }
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "status": "completed",
                "example": "04_tiny_transformer",
                "mode": "without_sahasra",
                "parameter_count": count_parameters(params),
                "train_examples": int(dataset["train_x"].shape[0]),
                "val_examples": int(dataset["val_x"].shape[0]),
                "sample_text": sample_text(
                    params,
                    prompt=args.prompt,
                    stoi=dataset["stoi"],
                    itos=dataset["itos"],
                    context_length=args.context_length,
                    sample_steps=args.sample_steps,
                    seed=args.seed + 99,
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
