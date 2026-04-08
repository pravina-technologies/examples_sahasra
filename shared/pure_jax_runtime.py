from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from sahasra_core.models import RemoteTensorHandle


def init_demo_params() -> dict[str, Any]:
    return {
        "encoder": {
            "w": jnp.ones((4, 3), dtype=jnp.float32),
            "b": jnp.zeros((3,), dtype=jnp.float32),
        },
        "head": {
            "w": jnp.full((3, 2), 2.0, dtype=jnp.float32),
            "b": jnp.ones((2,), dtype=jnp.float32),
        },
    }


def predict(params, x, mode: str = "eval"):
    hidden = jnp.tanh(x @ params["encoder"]["w"] + params["encoder"]["b"])
    logits = hidden @ params["head"]["w"] + params["head"]["b"]
    if mode == "train":
        logits = logits + 1.0
    return {
        "logits": logits,
        "stats": {
            "mean": jnp.mean(logits, axis=0),
            "max": jnp.max(logits, axis=0),
        },
    }


def predict_logits_only(params, x):
    hidden = jnp.tanh(x @ params["encoder"]["w"] + params["encoder"]["b"])
    return hidden @ params["head"]["w"] + params["head"]["b"]


def summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "logits_shape": [int(dim) for dim in output["logits"].shape],
        "logits_mean": float(np.asarray(output["logits"]).mean()),
        "stats_mean_shape": [int(dim) for dim in output["stats"]["mean"].shape],
        "stats_max_shape": [int(dim) for dim in output["stats"]["max"].shape],
        "stats_mean_sum": float(np.asarray(output["stats"]["mean"]).sum()),
        "stats_max_sum": float(np.asarray(output["stats"]["max"]).sum()),
    }


def describe_remote_tree(value: Any) -> Any:
    if isinstance(value, RemoteTensorHandle):
        return {
            "type": "RemoteTensorHandle",
            "shape": value.shape,
            "dtype": value.dtype,
        }
    if isinstance(value, dict):
        return {key: describe_remote_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [describe_remote_tree(child) for child in value]
    if isinstance(value, tuple):
        return [describe_remote_tree(child) for child in value]
    return {"type": type(value).__name__}


def summarize_params(params: dict[str, Any]) -> dict[str, float]:
    return {
        "encoder_w_sum": float(np.asarray(params["encoder"]["w"]).sum()),
        "head_b_sum": float(np.asarray(params["head"]["b"]).sum()),
    }

