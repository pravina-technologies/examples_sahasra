from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.pure_jax_runtime import describe_remote_tree, init_demo_params, predict


def main() -> None:
    params = init_demo_params()
    x = jnp.ones((2, 4), dtype=jnp.float32)

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        remote_params = sahasra.device_put(params, runtime=runtime)

        @runtime.jit(static_argnames=("mode",), output_mode="remote")
        def remote_predict(current_params, current_x, mode="eval"):
            return predict(current_params, current_x, mode=mode)

        result = remote_predict.remote(remote_params, x, mode="eval")
        remote_tree = result.output_tree
        materialized = sahasra.device_get(result)

        print(
            json.dumps(
                {
                    "example": "07_remote_output_tree",
                    "mode": "with_sahasra",
                    "remote_output_tree": describe_remote_tree(remote_tree),
                    "materialized_logits_shape": [int(dim) for dim in materialized["logits"].shape],
                    "materialized_stats_mean_shape": [int(dim) for dim in materialized["stats"]["mean"].shape],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
