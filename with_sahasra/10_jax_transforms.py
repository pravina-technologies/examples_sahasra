from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import sahasra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.pure_jax_runtime import init_demo_params, summarize_params


def model_apply(params, x):
    hidden = jnp.tanh(x @ params["encoder"]["w"] + params["encoder"]["b"])
    return hidden @ params["head"]["w"] + params["head"]["b"]


def loss_fn(params, x, y):
    preds = model_apply(params, x)
    return jnp.mean((preds - y) ** 2)


def grad_step(params, x, y, lr):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def value_and_grad_step(params, x, y, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return {"loss": loss, "params": new_params}


def vmap_predict(params, batch_x):
    return jax.vmap(lambda row: model_apply(params, row[None, :])[0])(batch_x)


def main() -> None:
    params = init_demo_params()
    x = jnp.ones((2, 4), dtype=jnp.float32)
    y = jnp.ones((2, 2), dtype=jnp.float32)
    batch_x = jnp.stack(
        [
            jnp.ones((4,), dtype=jnp.float32),
            jnp.full((4,), 2.0, dtype=jnp.float32),
            jnp.full((4,), -1.0, dtype=jnp.float32),
        ]
    )
    lr = 0.1

    local_grad = grad_step(params, x, y, lr)
    local_vg = value_and_grad_step(params, x, y, lr)
    local_vmap = vmap_predict(params, batch_x)

    with sahasra.SahasraRuntime.connect(
        base_url=os.getenv("SAHASRA_API_URL"),
        api_key=os.getenv("SAHASRA_API_BEARER_TOKEN"),
        gpu_class="g5",
        region="ap-south-1",
    ) as runtime:
        remote_params = sahasra.device_put(params, runtime=runtime)

        @runtime.jit
        def remote_grad_step(current_params, current_x, current_y, current_lr):
            return grad_step(current_params, current_x, current_y, current_lr)

        @runtime.jit
        def remote_value_and_grad_step(current_params, current_x, current_y, current_lr):
            return value_and_grad_step(current_params, current_x, current_y, current_lr)

        @runtime.jit
        def remote_vmap_predict(current_params, current_batch_x):
            return vmap_predict(current_params, current_batch_x)

        remote_grad = remote_grad_step(remote_params, x, y, lr)
        remote_vg = remote_value_and_grad_step(remote_params, x, y, lr)
        remote_vmap = remote_vmap_predict(remote_params, batch_x)

    print(
        json.dumps(
            {
                "example": "10_jax_transforms",
                "mode": "with_sahasra",
                "local_grad": summarize_params(local_grad),
                "remote_grad": summarize_params(remote_grad),
                "local_value_and_grad": {
                    "loss": float(np.asarray(local_vg["loss"])),
                    "params": summarize_params(local_vg["params"]),
                },
                "remote_value_and_grad": {
                    "loss": float(np.asarray(remote_vg["loss"])),
                    "params": summarize_params(remote_vg["params"]),
                },
                "local_vmap": {
                    "shape": [int(dim) for dim in local_vmap.shape],
                    "sum": float(np.asarray(local_vmap).sum()),
                },
                "remote_vmap": {
                    "shape": [int(dim) for dim in remote_vmap.shape],
                    "sum": float(np.asarray(remote_vmap).sum()),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
