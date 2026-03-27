from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.synthetic_mlp import init_params, logits, make_dataset


def main() -> None:
    import jax

    dataset = make_dataset(train_points=512, val_points=128, seed=3)
    params = init_params(jax.random.PRNGKey(0), hidden_dim=48)
    batch = dataset["val_x"]

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        remote_infer = sahasra.jit(logits, runtime=runtime, output_mode="remote")
        executions = []
        probs = None
        for _ in range(3):
            result = remote_infer.remote(params, batch)
            probs = np.asarray(result.materialize())
            executions.append(
                {
                    "id": result.execution.id,
                    "billing_status": result.execution.billing_status,
                    "billed_amount_inr": result.execution.billed_amount_inr,
                    "runtime_mode": result.execution.runtime_mode,
                }
            )

        print(
            json.dumps(
                {
                    "example": "02_mlp_inference",
                    "mode": "with_sahasra",
                    "batch_size": int(batch.shape[0]),
                    "output_shape": list(probs.shape),
                    "executions": executions,
                    "first_row": probs[0].round(4).tolist(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
