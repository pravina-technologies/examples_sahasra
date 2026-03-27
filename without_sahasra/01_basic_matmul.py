from __future__ import annotations

import json
import time

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
