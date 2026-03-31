from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_mnist_dataset(
    *,
    data_dir: Path = DATA_DIR,
    train_limit: int = 6000,
    val_limit: int = 1000,
) -> dict[str, np.ndarray]:
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError(
            "This example uses torchvision.datasets.MNIST. Install 'torch' and 'torchvision' first."
        ) from exc

    train_ds = MNIST(root=str(data_dir), train=True, download=True)
    test_ds = MNIST(root=str(data_dir), train=False, download=True)

    train_images = np.asarray(train_ds.data[:train_limit], dtype=np.float32)
    train_labels = np.asarray(train_ds.targets[:train_limit], dtype=np.int32)
    val_images = np.asarray(test_ds.data[:val_limit], dtype=np.float32)
    val_labels = np.asarray(test_ds.targets[:val_limit], dtype=np.int32)

    train_images = ((train_images / 255.0) - MNIST_MEAN) / MNIST_STD
    val_images = ((val_images / 255.0) - MNIST_MEAN) / MNIST_STD

    return {
        "train_x": train_images[..., None],
        "train_y": train_labels,
        "val_x": val_images[..., None],
        "val_y": val_labels,
    }


def init_params(
    key: jax.Array,
    *,
    conv1_channels: int = 16,
    conv2_channels: int = 32,
    hidden_dim: int = 128,
) -> dict[str, Any]:
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def randn(rng: jax.Array, shape: tuple[int, ...], scale: float) -> np.ndarray:
        return np.asarray(jax.random.normal(rng, shape, dtype=jnp.float32) * scale, dtype=np.float32)

    flattened_dim = 7 * 7 * conv2_channels
    return {
        "conv1": {
            "w": randn(k1, (3, 3, 1, conv1_channels), 0.08),
            "b": np.zeros((conv1_channels,), dtype=np.float32),
        },
        "conv2": {
            "w": randn(k2, (3, 3, conv1_channels, conv2_channels), 0.08),
            "b": np.zeros((conv2_channels,), dtype=np.float32),
        },
        "dense1": {
            "w": randn(k3, (flattened_dim, hidden_dim), 0.06),
            "b": np.zeros((hidden_dim,), dtype=np.float32),
        },
        "dense2": {
            "w": randn(k4, (hidden_dim, 10), 0.06),
            "b": np.zeros((10,), dtype=np.float32),
        },
    }


def conv2d(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    out = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return out + b


def avg_pool(x: jax.Array) -> jax.Array:
    batch, height, width, channels = x.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"avg_pool expects even spatial dimensions, got {(height, width)}")
    x = x.reshape(batch, height // 2, 2, width // 2, 2, channels)
    return jnp.mean(x, axis=(2, 4))


def logits(params: dict[str, Any], images: jax.Array) -> jax.Array:
    x = jax.nn.relu(conv2d(images, params["conv1"]["w"], params["conv1"]["b"]))
    x = avg_pool(x)
    x = jax.nn.relu(conv2d(x, params["conv2"]["w"], params["conv2"]["b"]))
    x = avg_pool(x)
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.relu(x @ params["dense1"]["w"] + params["dense1"]["b"])
    return x @ params["dense2"]["w"] + params["dense2"]["b"]


def loss_and_accuracy(
    params: dict[str, Any],
    images: jax.Array,
    labels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    scores = logits(params, images)
    log_probs = jax.nn.log_softmax(scores, axis=-1)
    loss = -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])
    predictions = jnp.argmax(scores, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return loss, accuracy


def train_step(
    params: dict[str, Any],
    images: jax.Array,
    labels: jax.Array,
    learning_rate: jax.Array,
) -> tuple[dict[str, Any], jax.Array, jax.Array]:
    def objective(current_params: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
        return loss_and_accuracy(current_params, images, labels)

    (loss, accuracy), grads = jax.value_and_grad(objective, has_aux=True)(params)
    next_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return next_params, loss, accuracy


def iterate_minibatches(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
):
    order = rng.permutation(images.shape[0])
    for start in range(0, images.shape[0], batch_size):
        idx = order[start : start + batch_size]
        yield images[idx], labels[idx]


def count_parameters(params: dict[str, Any]) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.asarray(leaf).size for leaf in leaves))


def summary_dict(dataset: dict[str, np.ndarray], params: dict[str, Any]) -> dict[str, int]:
    return {
        "train_examples": int(dataset["train_x"].shape[0]),
        "val_examples": int(dataset["val_x"].shape[0]),
        "parameter_count": count_parameters(params),
    }
