from __future__ import annotations

import json
import os
import time

import numpy as np
import sahasra


def main() -> None:
    a = np.linspace(-1.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    b = np.linspace(1.0, -1.0, 512 * 512, dtype=np.float32).reshape(512, 512)

    def matmul_relu(x, y):
        import jax.numpy as jnp

        return jnp.maximum(x @ y, 0.0)

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        remote_fn = sahasra.jit(matmul_relu, runtime=runtime, output_mode="remote")
        start = time.perf_counter()
        result = remote_fn.remote(a, b)
        out = result.materialize()
        elapsed = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "example": "01_basic_matmul",
                    "mode": "with_sahasra",
                    "shape": list(out.shape),
                    "mean": float(out.mean()),
                    "max": float(out.max()),
                    "runtime_mode": result.execution.runtime_mode,
                    "billing_status": result.execution.billing_status,
                    "billed_amount_inr": result.execution.billed_amount_inr,
                    "customer_summary": getattr(result.execution, "customer_summary", None),
                    "elapsed_sec": elapsed,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
