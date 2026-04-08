from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.pure_jax_runtime import describe_remote_tree, init_demo_params, predict


def main() -> None:
    params = init_demo_params()
    inputs = [
        jnp.ones((2, 4), dtype=jnp.float32),
        jnp.full((2, 4), 2.0, dtype=jnp.float32),
        jnp.full((2, 4), -1.0, dtype=jnp.float32),
    ]

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        remote_params = sahasra.device_put(params, runtime=runtime)

        @runtime.jit
        def materialized_predict(current_params, current_x):
            return predict(current_params, current_x)["logits"]

        @runtime.jit(output_mode="remote")
        def remote_predict(current_params, current_x):
            output = predict(current_params, current_x)
            return {
                "logits": output["logits"],
                "stats": {"sum": jnp.sum(current_x, axis=0)},
            }

        runs = []
        started = time.perf_counter()
        for run_index, current_x in enumerate(inputs, start=1):
            run_started = time.perf_counter()
            output = materialized_predict(remote_params, current_x)
            runs.append(
                {
                    "run_index": run_index,
                    "elapsed_sec": round(time.perf_counter() - run_started, 4),
                    "output_shape": [int(dim) for dim in output.shape],
                    "output_sum": float(np.asarray(output).sum()),
                }
            )
        repeated_elapsed = time.perf_counter() - started

        remote_result = remote_predict.remote(remote_params, inputs[0])
        remote_output_tree = remote_result.output_tree
        materialized_remote_output = sahasra.device_get(remote_result)

    print(
        json.dumps(
            {
                "example": "09_repeated_inference",
                "mode": "with_sahasra",
                "params_pinned_once": True,
                "repeat_count": len(runs),
                "repeated_total_elapsed_sec": round(repeated_elapsed, 4),
                "runs": runs,
                "remote_output_tree": describe_remote_tree(remote_output_tree),
                "materialized_remote_logits_shape": [int(dim) for dim in materialized_remote_output["logits"].shape],
                "materialized_remote_stats_sum_shape": [
                    int(dim) for dim in materialized_remote_output["stats"]["sum"].shape
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
