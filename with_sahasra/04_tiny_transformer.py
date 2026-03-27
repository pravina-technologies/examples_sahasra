from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.tiny_transformer import (
    build_char_dataset,
    count_parameters,
    eval_step,
    init_params,
    sample_text,
    train_step,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote tiny transformer example with Sahasra.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--steps-per-execution", type=int, default=4)
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
    learning_rate = np.asarray(args.learning_rate, dtype=np.float32)
    shuffle_rng = np.random.default_rng(args.seed + 17)

    runtime = sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    )
    trainer = sahasra.trainer(
        train_fn=lambda state, x, y, lr: train_step(state, x, y, lr),
        eval_fn=lambda state, x, y: eval_step(state, x, y),
        runtime=runtime,
        chunk_arity=2,
        steps_per_execution=args.steps_per_execution,
    )
    val_x_ref, val_y_ref = runtime.pin_tensors([dataset["val_x"], dataset["val_y"]])

    state_refs = params
    final_params = None
    last_execution = None
    try:
        for epoch in range(1, args.epochs + 1):
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
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "train_loss": float(np.asarray(train_loss)),
                        "train_accuracy": float(np.asarray(train_accuracy)),
                        "val_loss": float(np.asarray(val_loss)),
                        "val_accuracy": float(np.asarray(val_accuracy)),
                        "runtime_mode": last_execution.runtime_mode,
                        "billing_status": last_execution.billing_status,
                        "billed_amount_inr": last_execution.billed_amount_inr,
                        "chunk_calls": epoch_result.chunk_calls,
                        "steps_per_execution": epoch_result.steps_per_execution,
                        "mode": "with_sahasra",
                    }
                ),
                flush=True,
            )

        final_params = epoch_result.epoch_result.materialize_state()
        print(
            json.dumps(
                {
                    "status": "completed",
                    "example": "04_tiny_transformer",
                    "mode": "with_sahasra",
                    "session_id": runtime.session.session.id,
                    "worker_id": runtime.session.session.worker_id,
                    "parameter_count": count_parameters(final_params),
                    "train_examples": int(dataset["train_x"].shape[0]),
                    "val_examples": int(dataset["val_x"].shape[0]),
                    "final_runtime_mode": last_execution.runtime_mode if last_execution is not None else None,
                    "final_customer_summary": getattr(last_execution, "customer_summary", None),
                    "sample_text": sample_text(
                        final_params,
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
