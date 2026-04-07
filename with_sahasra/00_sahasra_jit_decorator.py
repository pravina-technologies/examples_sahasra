from __future__ import annotations

import json
import os
import time

import numpy as np
import sahasra


@sahasra.jit(
    base_url=os.getenv("SAHASRA_API_URL"),
    api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
    gpu_class="g5",
    region="ap-south-1",
    output_mode="remote",
    static_argnames=("mode",),
)
def summarize_batch(x, mode="eval"):
    import jax.numpy as jnp

    shifted = x + 1.0 if mode == "train" else x - 1.0
    return {
        "shifted": shifted,
        "stats": {
            "mean": jnp.mean(shifted, axis=-1),
            "sum": jnp.sum(shifted, axis=-1),
        },
    }


def main() -> None:
    x = np.arange(8, dtype=np.float32).reshape(2, 4)

    start = time.perf_counter()
    result = summarize_batch.remote(x, mode="eval")
    materialized = sahasra.device_get(result)
    elapsed = time.perf_counter() - start

    print(
        json.dumps(
            {
                "example": "00_sahasra_jit_decorator",
                "mode": "with_sahasra",
                "decorator": "@sahasra.jit",
                "static_argnames": ["mode"],
                "input_shape": list(x.shape),
                "remote_output_tree_type": type(result.output_tree).__name__,
                "runtime_mode": result.execution.runtime_mode,
                "billing_status": result.execution.billing_status,
                "billed_amount_inr": result.execution.billed_amount_inr,
                "elapsed_sec": elapsed,
                "shifted_shape": list(np.asarray(materialized["shifted"]).shape),
                "stats_mean": np.asarray(materialized["stats"]["mean"]).round(4).tolist(),
                "stats_sum": np.asarray(materialized["stats"]["sum"]).round(4).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
