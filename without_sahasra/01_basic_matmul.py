from __future__ import annotations

import json
import os
import time

if os.getenv("SAHASRA_EXAMPLES_FORCE_CPU") == "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    a = np.linspace(-1.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    b = np.linspace(1.0, -1.0, 512 * 512, dtype=np.float32).reshape(512, 512)

    @jax.jit
    def matmul_relu(x, y):
        return jnp.maximum(x @ y, 0.0)

    start = time.perf_counter()
    out = np.asarray(matmul_relu(a, b))
    elapsed = time.perf_counter() - start

    print(
        json.dumps(
            {
                "example": "01_basic_matmul",
                "mode": "without_sahasra",
                "backend": jax.default_backend(),
                "shape": list(out.shape),
                "mean": float(out.mean()),
                "max": float(out.max()),
                "elapsed_sec": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
