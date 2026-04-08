from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.pure_jax_runtime import init_demo_params, predict_logits_only


def main() -> None:
    checkpoint_path = Path(__file__).resolve().parents[1] / "shared" / "checkpoints" / "phase11_roundtrip.npz"
    params = init_demo_params()
    x = jnp.ones((2, 4), dtype=jnp.float32)

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        remote_params = sahasra.device_put(params, runtime=runtime)

        @runtime.jit
        def remote_predict(current_params, current_x):
            return predict_logits_only(current_params, current_x)

        before = remote_predict(remote_params, x)
        sahasra.save_checkpoint(checkpoint_path, remote_params, runtime=runtime)

    loaded_params = sahasra.load_checkpoint(checkpoint_path)

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        remote_loaded = sahasra.device_put(loaded_params, runtime=runtime)

        @runtime.jit
        def remote_predict(current_params, current_x):
            return predict_logits_only(current_params, current_x)

        after = remote_predict(remote_loaded, x)

    print(
        json.dumps(
            {
                "example": "08_checkpoint_roundtrip",
                "mode": "with_sahasra",
                "checkpoint_path": str(checkpoint_path),
                "loaded_encoder_w_shape": [int(dim) for dim in loaded_params["encoder"]["w"].shape],
                "before_sum": float(np.asarray(before).sum()),
                "after_sum": float(np.asarray(after).sum()),
            },
            indent=2,
        )
    )

    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
