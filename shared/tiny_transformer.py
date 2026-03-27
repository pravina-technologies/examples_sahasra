from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


DATA_PATH = Path(__file__).resolve().parent / "data" / "corpus.txt"


def load_corpus(path: Path = DATA_PATH) -> str:
    return path.read_text(encoding="utf-8")


def build_char_dataset(
    *,
    corpus_path: Path = DATA_PATH,
    context_length: int = 48,
    stride: int = 6,
    train_ratio: float = 0.9,
) -> dict[str, Any]:
    text = load_corpus(corpus_path)
    vocab = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(vocab)}
    itos = {idx: ch for ch, idx in stoi.items()}
    tokens = np.asarray([stoi[ch] for ch in text], dtype=np.int32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    max_start = len(tokens) - context_length - 1
    for start in range(0, max_start + 1, stride):
        xs.append(tokens[start : start + context_length])
        ys.append(tokens[start + 1 : start + context_length + 1])

    x = np.stack(xs).astype(np.int32)
    y = np.stack(ys).astype(np.int32)
    split = max(1, min(len(x) - 1, int(len(x) * train_ratio)))

    return {
        "text": text,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(vocab),
        "train_x": x[:split],
        "train_y": y[:split],
        "val_x": x[split:],
        "val_y": y[split:],
        "context_length": context_length,
    }


def init_params(
    key: jax.Array,
    *,
    vocab_size: int,
    context_length: int,
    d_model: int = 48,
    n_heads: int = 4,
    mlp_dim: int = 96,
) -> dict[str, Any]:
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")

    keys = jax.random.split(key, 9)

    def randn(k: jax.Array, shape: tuple[int, ...], scale: float) -> np.ndarray:
        return np.asarray(jax.random.normal(k, shape, dtype=jnp.float32) * scale, dtype=np.float32)

    return {
        "token_embed": randn(keys[0], (vocab_size, d_model), 0.05),
        "pos_embed": randn(keys[1], (context_length, d_model), 0.02),
        "block": {
            "ln1_scale": np.ones((d_model,), dtype=np.float32),
            "ln1_bias": np.zeros((d_model,), dtype=np.float32),
            "wq": randn(keys[2], (d_model, d_model), 0.08),
            "wk": randn(keys[3], (d_model, d_model), 0.08),
            "wv": randn(keys[4], (d_model, d_model), 0.08),
            "wo": randn(keys[5], (d_model, d_model), 0.08),
            "ln2_scale": np.ones((d_model,), dtype=np.float32),
            "ln2_bias": np.zeros((d_model,), dtype=np.float32),
            "w1": randn(keys[6], (d_model, mlp_dim), 0.08),
            "b1": np.zeros((mlp_dim,), dtype=np.float32),
            "w2": randn(keys[7], (mlp_dim, d_model), 0.08),
            "b2": np.zeros((d_model,), dtype=np.float32),
        },
        "ln_f_scale": np.ones((d_model,), dtype=np.float32),
        "ln_f_bias": np.zeros((d_model,), dtype=np.float32),
        "lm_head": randn(keys[8], (d_model, vocab_size), 0.05),
        "lm_head_bias": np.zeros((vocab_size,), dtype=np.float32),
    }


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array, eps: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    return normalized * scale + bias


def causal_attention(x: jax.Array, params: dict[str, jax.Array], n_heads: int) -> jax.Array:
    batch, steps, width = x.shape
    head_dim = width // n_heads
    q = x @ params["wq"]
    k = x @ params["wk"]
    v = x @ params["wv"]

    q = q.reshape(batch, steps, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, steps, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, steps, n_heads, head_dim).transpose(0, 2, 1, 3)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
    mask = jnp.tril(jnp.ones((steps, steps), dtype=bool))
    scores = jnp.where(mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    attended = attended.transpose(0, 2, 1, 3).reshape(batch, steps, width)
    return attended @ params["wo"]


def forward(params: dict[str, Any], x_tokens: jax.Array, *, n_heads: int = 4) -> jax.Array:
    _, steps = x_tokens.shape
    x = params["token_embed"][x_tokens] + params["pos_embed"][:steps][None, :, :]
    block = params["block"]

    h = layer_norm(x, block["ln1_scale"], block["ln1_bias"])
    x = x + causal_attention(h, block, n_heads)

    h2 = layer_norm(x, block["ln2_scale"], block["ln2_bias"])
    mlp = jax.nn.gelu(h2 @ block["w1"] + block["b1"])
    mlp = mlp @ block["w2"] + block["b2"]
    x = x + mlp

    x = layer_norm(x, params["ln_f_scale"], params["ln_f_bias"])
    return x @ params["lm_head"] + params["lm_head_bias"]


def loss_and_accuracy(
    params: dict[str, Any],
    x_tokens: jax.Array,
    y_tokens: jax.Array,
    *,
    n_heads: int = 4,
) -> tuple[jax.Array, jax.Array]:
    logits = forward(params, x_tokens, n_heads=n_heads)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    chosen = jnp.take_along_axis(log_probs, y_tokens[..., None], axis=-1)[..., 0]
    loss = -jnp.mean(chosen)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == y_tokens)
    return loss, accuracy


def train_step(
    params: dict[str, Any],
    x_tokens: jax.Array,
    y_tokens: jax.Array,
    learning_rate: jax.Array,
    *,
    n_heads: int = 4,
) -> tuple[dict[str, Any], jax.Array, jax.Array]:
    def objective(current_params: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
        return loss_and_accuracy(current_params, x_tokens, y_tokens, n_heads=n_heads)

    (loss, accuracy), grads = jax.value_and_grad(objective, has_aux=True)(params)
    next_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return next_params, loss, accuracy


def eval_step(params: dict[str, Any], x_tokens: jax.Array, y_tokens: jax.Array, *, n_heads: int = 4):
    return loss_and_accuracy(params, x_tokens, y_tokens, n_heads=n_heads)


def iterate_minibatches(
    x_tokens: np.ndarray,
    y_tokens: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
):
    order = rng.permutation(x_tokens.shape[0])
    for start in range(0, x_tokens.shape[0], batch_size):
        idx = order[start : start + batch_size]
        yield x_tokens[idx], y_tokens[idx]


def encode_text(text: str, stoi: dict[str, int]) -> list[int]:
    fallback = stoi.get(" ", 0)
    return [stoi.get(ch, fallback) for ch in text]


def decode_tokens(tokens: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[int(token)] for token in tokens)


def sample_text(
    params: dict[str, Any],
    *,
    prompt: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    context_length: int,
    sample_steps: int = 80,
    temperature: float = 0.9,
    seed: int = 0,
    n_heads: int = 4,
) -> str:
    rng = np.random.default_rng(seed)
    tokens = encode_text(prompt, stoi)
    if not tokens:
        tokens = [0]

    for _ in range(sample_steps):
        context = tokens[-context_length:]
        x = np.asarray(context, dtype=np.int32)[None, :]
        logits = np.asarray(forward(params, x, n_heads=n_heads))[0, -1]
        logits = logits / max(temperature, 1e-4)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        tokens.append(int(rng.choice(len(probs), p=probs)))

    return decode_tokens(tokens, itos)


def count_parameters(params: dict[str, Any]) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.asarray(leaf).size for leaf in leaves))
