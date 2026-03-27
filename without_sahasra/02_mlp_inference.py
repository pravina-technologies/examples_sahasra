from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.synthetic_mlp import init_params, logits, make_dataset


def main() -> None:
    dataset = make_dataset(train_points=512, val_points=128, seed=3)
    params = init_params(jax.random.PRNGKey(0), hidden_dim=48)
    batch = dataset["val_x"]

    @jax.jit
    def infer(current_params, x):
        return logits(current_params, x)

    times = []
    probs = None
    for _ in range(5):
        start = time.perf_counter()
        scores = infer(params, batch)
        probs = np.asarray(jax.nn.softmax(scores, axis=-1))
        times.append(time.perf_counter() - start)

    print(
        json.dumps(
            {
                "example": "02_mlp_inference",
                "mode": "without_sahasra",
                "batch_size": int(batch.shape[0]),
                "output_shape": list(probs.shape),
                "avg_elapsed_sec": float(np.mean(times)),
                "first_row_probs": probs[0].round(4).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
