from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.pure_jax_runtime import init_demo_params, predict, summarize_output


def main() -> None:
    params = init_demo_params()
    x = jnp.ones((2, 4), dtype=jnp.float32)
    local_eval = predict(params, x, mode="eval")

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        remote_params = sahasra.device_put(params, runtime=runtime)

        @runtime.jit(static_argnames=("mode",))
        def remote_predict(current_params, current_x, mode="eval"):
            return predict(current_params, current_x, mode=mode)

        remote_eval = remote_predict(remote_params, x, mode="eval")
        remote_train = remote_predict(remote_params, x, mode="train")

        print(
            json.dumps(
                {
                    "example": "06_pytree_inference",
                    "mode": "with_sahasra",
                    "local_eval": summarize_output(local_eval),
                    "remote_eval": summarize_output(remote_eval),
                    "remote_train": summarize_output(remote_train),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
