from __future__ import annotations

import json
import os
import time
import sys
from pathlib import Path

if os.getenv("SAHASRA_EXAMPLES_FORCE_CPU") == "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.synthetic_mlp import init_params, logits, make_dataset


def infer_logits(w1, b1, w2, b2, x):
    return logits({"w1": w1, "b1": b1, "w2": w2, "b2": b2}, x)


def main() -> None:
    dataset = make_dataset(train_points=512, val_points=128, seed=3)
    params = init_params(jax.random.PRNGKey(0), hidden_dim=48)
    batch = dataset["val_x"]

    @jax.jit
    def infer(w1, b1, w2, b2, x):
        return infer_logits(w1, b1, w2, b2, x)

    times = []
    probs = None
    for _ in range(5):
        start = time.perf_counter()
        scores = infer(params["w1"], params["b1"], params["w2"], params["b2"], batch)
        probs = np.asarray(jax.nn.softmax(scores, axis=-1))
        times.append(time.perf_counter() - start)

    print(
        json.dumps(
            {
                "example": "02_mlp_inference",
                "mode": "without_sahasra",
                "backend": jax.default_backend(),
                "batch_size": int(batch.shape[0]),
                "output_shape": list(probs.shape),
                "elapsed_sec": [round(float(value), 6) for value in times],
                "avg_elapsed_sec": float(np.mean(times)),
                "first_row_probs": probs[0].round(4).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
