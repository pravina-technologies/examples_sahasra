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

from shared.synthetic_mlp import init_params, make_dataset, summary_dict, train_step, loss_and_accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote MLP training example with Sahasra.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--steps-per-execution", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = make_dataset(seed=args.seed)
    params = init_params(jax.random.PRNGKey(args.seed), hidden_dim=32)
    learning_rate = np.asarray(args.learning_rate, dtype=np.float32)
    shuffle_rng = np.random.default_rng(args.seed + 13)

    runtime = sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
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
    final_state = None
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
            (train_loss, train_accuracy), (val_loss, val_accuracy) = epoch_result.materialize_metrics()
            execution = epoch_result.epoch_result.execution
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "train_loss": float(np.asarray(train_loss)),
                        "train_accuracy": float(np.asarray(train_accuracy)),
                        "val_loss": float(np.asarray(val_loss)),
                        "val_accuracy": float(np.asarray(val_accuracy)),
                        "runtime_mode": execution.runtime_mode,
                        "billing_status": execution.billing_status,
                        "billed_amount_inr": execution.billed_amount_inr,
                        "mode": "with_sahasra",
                    }
                ),
                flush=True,
            )
        final_state = epoch_result.epoch_result.materialize_state()
        print(
            json.dumps(
                {
                    "status": "completed",
                    "example": "03_mlp_training",
                    "mode": "with_sahasra",
                    "session_id": runtime.session.session.id,
                    "worker_id": runtime.session.session.worker_id,
                    **summary_dict(dataset, final_state),
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
