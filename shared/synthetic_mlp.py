from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def make_dataset(
    *,
    train_points: int = 512,
    val_points: int = 128,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def sample_split(points: int) -> tuple[np.ndarray, np.ndarray]:
        per_class = points // 2
        class0 = rng.normal(loc=(-1.25, -0.9), scale=(0.45, 0.55), size=(per_class, 2))
        class1 = rng.normal(loc=(1.15, 1.05), scale=(0.5, 0.45), size=(per_class, 2))
        x = np.concatenate([class0, class1], axis=0).astype(np.float32)
        y = np.concatenate(
            [
                np.zeros((per_class,), dtype=np.int32),
                np.ones((per_class,), dtype=np.int32),
            ],
            axis=0,
        )
        order = rng.permutation(x.shape[0])
        return x[order], y[order]

    train_x, train_y = sample_split(train_points)
    val_x, val_y = sample_split(val_points)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
    }


def init_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
) -> dict[str, np.ndarray]:
    k1, k2 = jax.random.split(key)
    return {
        "w1": np.asarray(jax.random.normal(k1, (2, hidden_dim), dtype=jnp.float32) * 0.25),
        "b1": np.zeros((hidden_dim,), dtype=np.float32),
        "w2": np.asarray(jax.random.normal(k2, (hidden_dim, 2), dtype=jnp.float32) * 0.25),
        "b2": np.zeros((2,), dtype=np.float32),
    }


def logits(params: dict[str, Any], x: jax.Array) -> jax.Array:
    hidden = jax.nn.gelu(x @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def loss_and_accuracy(params: dict[str, Any], x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
    scores = logits(params, x)
    log_probs = jax.nn.log_softmax(scores, axis=-1)
    loss = -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])
    predictions = jnp.argmax(scores, axis=-1)
    accuracy = jnp.mean(predictions == y)
    return loss, accuracy


def train_step(
    params: dict[str, Any],
    x: jax.Array,
    y: jax.Array,
    learning_rate: jax.Array,
) -> tuple[dict[str, Any], jax.Array, jax.Array]:
    def objective(current_params: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
        return loss_and_accuracy(current_params, x, y)

    (loss, accuracy), grads = jax.value_and_grad(objective, has_aux=True)(params)
    next_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return next_params, loss, accuracy


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
):
    order = rng.permutation(x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def summary_dict(dataset: dict[str, np.ndarray], params: dict[str, Any]) -> dict[str, int]:
    leaves = jax.tree_util.tree_leaves(params)
    return {
        "train_examples": int(dataset["train_x"].shape[0]),
        "val_examples": int(dataset["val_x"].shape[0]),
        "parameter_count": int(sum(np.asarray(leaf).size for leaf in leaves)),
    }
