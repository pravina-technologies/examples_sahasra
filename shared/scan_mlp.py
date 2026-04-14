from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def make_gaussian_mixture_dataset(
    *,
    train_examples: int = 2048,
    val_examples: int = 512,
    num_classes: int = 10,
    feature_dim: int = 64,
    class_separation: float = 2.25,
    noise_scale: float = 1.15,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build a reproducible multi-class synthetic dataset.

    The classes are intentionally learnable but not entirely trivial. This keeps
    the example self-contained while still exercising real mini-batch training.
    """
    rng = np.random.default_rng(seed)
    total_examples = train_examples + val_examples
    per_class = int(np.ceil(total_examples / num_classes))
    class_means = rng.standard_normal((num_classes, feature_dim)).astype(np.float32) * class_separation

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for class_id in range(num_classes):
        xs.append(
            (
                rng.standard_normal((per_class, feature_dim)).astype(np.float32) * noise_scale
                + class_means[class_id]
            )
        )
        ys.append(np.full((per_class,), class_id, dtype=np.int32))

    x_all = np.concatenate(xs, axis=0)[:total_examples]
    y_all = np.concatenate(ys, axis=0)[:total_examples]
    order = rng.permutation(total_examples)
    x_all = x_all[order].astype(np.float32)
    y_all = y_all[order].astype(np.int32)

    return {
        "train_x": x_all[:train_examples],
        "train_y": y_all[:train_examples],
        "val_x": x_all[train_examples:],
        "val_y": y_all[train_examples:],
    }


def init_params(
    key: jax.Array,
    *,
    feature_dim: int = 64,
    hidden1: int = 256,
    hidden2: int = 128,
    num_classes: int = 10,
) -> dict[str, np.ndarray]:
    k1, k2, k3 = jax.random.split(key, 3)

    def he_scale(fan_in: int) -> float:
        return float(np.sqrt(2.0 / fan_in))

    return {
        "w1": np.asarray(jax.random.normal(k1, (feature_dim, hidden1), dtype=jnp.float32) * he_scale(feature_dim)),
        "b1": np.zeros((hidden1,), dtype=np.float32),
        "w2": np.asarray(jax.random.normal(k2, (hidden1, hidden2), dtype=jnp.float32) * he_scale(hidden1)),
        "b2": np.zeros((hidden2,), dtype=np.float32),
        "w3": np.asarray(jax.random.normal(k3, (hidden2, num_classes), dtype=jnp.float32) * he_scale(hidden2)),
        "b3": np.zeros((num_classes,), dtype=np.float32),
    }


def forward(params: dict[str, Any], x: jax.Array) -> jax.Array:
    hidden = jax.nn.relu(x @ params["w1"] + params["b1"])
    hidden = jax.nn.relu(hidden @ params["w2"] + params["b2"])
    return hidden @ params["w3"] + params["b3"]


def cross_entropy(params: dict[str, Any], x: jax.Array, y: jax.Array) -> jax.Array:
    scores = forward(params, x)
    log_probs = jax.nn.log_softmax(scores, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])


def accuracy(params: dict[str, Any], x: jax.Array, y: jax.Array) -> jax.Array:
    predictions = jnp.argmax(forward(params, x), axis=-1)
    return jnp.mean(predictions == y)


def init_adam(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    zeros = jax.tree_util.tree_map(lambda leaf: np.zeros_like(np.asarray(leaf), dtype=np.float32), params)
    return zeros, zeros, np.asarray(0.0, dtype=np.float32)


def adam_step(
    params: dict[str, Any],
    grads: dict[str, Any],
    opt_state: tuple[dict[str, Any], dict[str, Any], jax.Array],
    learning_rate: jax.Array,
) -> tuple[dict[str, Any], tuple[dict[str, Any], dict[str, Any], jax.Array]]:
    beta1 = jnp.asarray(0.9, dtype=jnp.float32)
    beta2 = jnp.asarray(0.999, dtype=jnp.float32)
    eps = jnp.asarray(1e-8, dtype=jnp.float32)
    m, v, step = opt_state
    step = step + jnp.asarray(1.0, dtype=jnp.float32)
    m = jax.tree_util.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, m, grads)
    v = jax.tree_util.tree_map(lambda old, grad: beta2 * old + (1.0 - beta2) * (grad * grad), v, grads)
    m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**step), v)
    next_params = jax.tree_util.tree_map(
        lambda param, mh, vh: param - learning_rate * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return next_params, (m, v, step)


def build_batch_indices(
    *,
    train_examples: int,
    batch_size: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    steps_per_epoch = train_examples // batch_size
    rng = np.random.default_rng(seed)
    batch_indices = np.empty((epochs * steps_per_epoch, batch_size), dtype=np.int32)
    for epoch in range(epochs):
        order = rng.permutation(train_examples)
        for step in range(steps_per_epoch):
            start = step * batch_size
            batch_indices[epoch * steps_per_epoch + step] = order[start : start + batch_size]
    return batch_indices


def train_many(
    params: dict[str, Any],
    m: dict[str, Any],
    v: dict[str, Any],
    step: jax.Array,
    train_x: jax.Array,
    train_y: jax.Array,
    batch_indices: jax.Array,
    learning_rate: jax.Array,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], jax.Array, jax.Array]:
    """Run all mini-batch Adam updates inside one JAX scan."""

    def one_step(carry, batch_index):
        current_params, current_m, current_v, current_step = carry
        batch_x = train_x[batch_index]
        batch_y = train_y[batch_index]
        loss, grads = jax.value_and_grad(cross_entropy)(current_params, batch_x, batch_y)
        next_params, (next_m, next_v, next_step) = adam_step(
            current_params,
            grads,
            (current_m, current_v, current_step),
            learning_rate,
        )
        return (next_params, next_m, next_v, next_step), loss

    (final_params, final_m, final_v, final_step), losses = jax.lax.scan(
        one_step,
        (params, m, v, step),
        batch_indices,
    )
    return final_params, final_m, final_v, final_step, losses


def confusion_matrix(
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    predictions = np.asarray(jnp.argmax(forward(params, x), axis=-1))
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for target, prediction in zip(y, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def parameter_count(params: dict[str, Any]) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)))

