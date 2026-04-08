# Sahasra Examples

This repo is a public companion repo for **Sahasra**.

It is organized to make one idea very easy to understand:

**the same kind of JAX code can be run locally or through Sahasra with only small changes.**

So the repo is split into two parallel folders:

- [without_sahasra](./without_sahasra): plain local JAX examples
- [with_sahasra](./with_sahasra): the same style of examples adapted to run on Sahasra

The examples are ordered from basic to more demanding:

0. `00_sahasra_jit_decorator.py` in `with_sahasra/`
1. `01_basic_matmul.py`
2. `02_mlp_inference.py`
3. `03_mlp_training.py`
4. `04_tiny_transformer.py`
5. `05_mnist_cnn.py`
6. `05_mnist_cnn_inference.py` in `with_sahasra/`

There is also an advanced pure-JAX Sahasra-only set that mirrors the latest live-validated Phase 11 features:

7. `06_pytree_inference.py`
8. `07_remote_output_tree.py`
9. `08_checkpoint_roundtrip.py`
10. `09_repeated_inference.py`
11. `10_jax_transforms.py`

The newest example pair uses a real dataset:

- `05_mnist_cnn.py` downloads MNIST through `torchvision.datasets.MNIST`
- the local and Sahasra versions share the same CNN model code
- both print richer progress so you can watch training and validation improve live
- the Sahasra training script saves a `.npz` checkpoint, and `05_mnist_cnn_inference.py` reloads it in a fresh inference session

## What This Repo Is For

This repo is for people who want to answer questions like:

- What changes when I move a local JAX example to Sahasra?
- How small can the code diff be?
- How do warm sessions and `SahasraRuntime` fit into repeated work?
- How does remote training look compared to local training?

It is not trying to hide the difference between local and remote execution.
It is trying to make that difference understandable and practical.

## Quick Start

If you want the fastest path through the repo:

1. Run one local example from `without_sahasra/`
2. Install Sahasra `0.1.2` from TestPyPI
3. Create an API key from the Sahasra app
4. Run the matching example from `with_sahasra/`

Suggested first pair:

```bash
python without_sahasra/01_basic_matmul.py
python with_sahasra/01_basic_matmul.py
```

If you specifically want to see the decorator-style public API first:

```bash
python with_sahasra/00_sahasra_jit_decorator.py
```

If you want the newest pure-JAX Sahasra behavior after that, use this order:

```bash
python with_sahasra/06_pytree_inference.py
python with_sahasra/07_remote_output_tree.py
python with_sahasra/08_checkpoint_roundtrip.py
python with_sahasra/09_repeated_inference.py
python with_sahasra/10_jax_transforms.py
```

## Install

### Local-only JAX examples

If you only want to run the `without_sahasra/` examples:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "jax>=0.4.35" "numpy>=1.26" "torch>=2.8" "torchvision>=0.23"
```

### Sahasra examples from TestPyPI

If you want to run the `with_sahasra/` examples:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install \
  --index-url https://pypi.org/simple \
  "httpx>=0.27,<1" \
  "pydantic>=2.9,<3" \
  "flatbuffers>=24.3.25" \
  "jax>=0.4.35"

python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  "sahasra-core==0.1.2"

python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  "sahasra==0.1.2"
```

Then configure your Sahasra environment:

```bash
export SAHASRA_API_URL="https://demo.sahasra.dev"
export SAHASRA_API_BEARER_TOKEN="sk_sahasra_..."
```

Check that your install works:

```bash
sahasra doctor --base-url https://demo.sahasra.dev
sahasra me --base-url https://demo.sahasra.dev
sahasra billing-me --base-url https://demo.sahasra.dev
```

Current validated invite-only beta packages:

- `sahasra-core==0.1.2`
- `sahasra==0.1.2`

## Run The Examples

### Local

```bash
python without_sahasra/01_basic_matmul.py
python without_sahasra/02_mlp_inference.py
python without_sahasra/03_mlp_training.py
python without_sahasra/04_tiny_transformer.py --epochs 2
python without_sahasra/05_mnist_cnn.py --epochs 3
```

### Sahasra

```bash
python with_sahasra/01_basic_matmul.py
python with_sahasra/02_mlp_inference.py
python with_sahasra/03_mlp_training.py --epochs 2 --steps-per-execution 8
python with_sahasra/04_tiny_transformer.py --epochs 2 --steps-per-execution 8
python with_sahasra/05_mnist_cnn.py --epochs 3 --steps-per-execution 8
python with_sahasra/05_mnist_cnn_inference.py
python with_sahasra/06_pytree_inference.py
python with_sahasra/07_remote_output_tree.py
python with_sahasra/08_checkpoint_roundtrip.py
python with_sahasra/09_repeated_inference.py
python with_sahasra/10_jax_transforms.py
```

## Timing Output

The examples now print timing information directly in their JSON output so local and remote runs are easier to compare:

- `01_basic_matmul.py`
  - `elapsed_sec`
- `02_mlp_inference.py`
  - `elapsed_sec`
  - `avg_elapsed_sec`
- `03_mlp_training.py`
  - `epoch_elapsed_sec`
  - `total_elapsed_sec`
- `04_tiny_transformer.py`
  - `epoch_elapsed_sec`
  - `total_elapsed_sec`
- `05_mnist_cnn.py`
  - `epoch_elapsed_sec`
  - `total_elapsed_sec`
  - `final_val_accuracy`
  - with Sahasra also prints `total_billed_inr`
- `05_mnist_cnn_inference.py`
  - `predict_elapsed_sec`
  - `eval_accuracy`
  - `sample_predictions`
- `06_pytree_inference.py`
  - local vs remote nested pytree output summaries
- `07_remote_output_tree.py`
  - remote output tree structure and materialized shapes
- `08_checkpoint_roundtrip.py`
  - checkpoint path and before/after output equality
- `09_repeated_inference.py`
  - repeated warm-runtime timings and remote output tree structure
- `10_jax_transforms.py`
  - local vs remote `grad`, `value_and_grad`, and `vmap` summaries

For local runs, the examples also print `backend`, so you can see whether JAX is using `cpu` or `gpu`.

### Screenshot / CPU-only mode

If you want the local examples to stay on CPU for clean screenshots or CPU-only comparisons:

```bash
export SAHASRA_EXAMPLES_FORCE_CPU=1
python without_sahasra/04_tiny_transformer.py
unset SAHASRA_EXAMPLES_FORCE_CPU
```

### Local GPU JAX

If your machine has a supported NVIDIA driver, you can install GPU JAX and run the local examples on GPU:

```bash
python -m pip install --upgrade "jax[cuda13]"
```

The examples also set:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

to reduce aggressive GPU memory preallocation during local runs.

Downloaded MNIST files are cached locally under `shared/data/mnist/` and are ignored by git.
Saved CNN checkpoints default to `shared/checkpoints/` and are ignored by git.

## Latest Timing Snapshot

Latest timings collected on the current demo setup:

- local runs: direct JAX on the RTX 5070 Ti host with CUDA JAX enabled
- Sahasra runs: the same host's RTX 5070 Ti reached through the Sahasra API/runtime path
- so these numbers are best read as a workflow/product comparison, not a hardware comparison

| Example | Local | With Sahasra | Notes |
| --- | --- | --- | --- |
| `01_basic_matmul.py` | `0.475s` | `9.515s` | remote path includes end-to-end runtime/orchestration overhead for a tiny workload |
| `02_mlp_inference.py` | `0.075s` avg across 5 runs | `4.915s` avg across 3 runs | warm repeated runs get cheaper on both sides |
| `03_mlp_training.py` | `1.407s` total | `207.521s` total | historical remote snapshot used `sahasra.trainer(..., steps_per_execution=2)` |
| `04_tiny_transformer.py` | `7.893s` total | `90.289s` total | historical remote snapshot used `sahasra.trainer(..., steps_per_execution=2)` |

These are not meant to claim that Sahasra is faster than local execution on small jobs.
They are meant to show the current product tradeoff clearly:

- local direct JAX is still much faster for small workloads on the same GPU host
- Sahasra adds control-plane and materialization overhead
- the benefit is the remote execution product surface: API key auth, wallet billing, reusable sessions, remote tensor handling, and the ability to move work off the laptop/workstation workflow

## Latest Local GPU Baseline

Latest local baseline collected on the RTX 5070 Ti host with CUDA JAX enabled:

| Example | Mode | Key Timing |
| --- | --- | --- |
| `01_basic_matmul.py` | local GPU | `elapsed_sec = 0.475s` |
| `02_mlp_inference.py` | local GPU | `avg_elapsed_sec = 0.075s` across 5 runs |
| `03_mlp_training.py` | local GPU | `total_elapsed_sec = 1.407s` |
| `04_tiny_transformer.py` | local GPU | `total_elapsed_sec = 7.893s` for 2 epochs |

Notes:

- first-run JAX compile/warmup dominates several local timings
- later epochs can be much faster than the first epoch once kernels are warm
- the tiny transformer local GPU run still showed first-run CUDA/JAX warnings on this host, but it completed successfully and reported GPU timings

## Latest Sahasra Remote Snapshot

Latest Sahasra timings collected against `https://demo.sahasra.dev` on the current single-worker beta:

| Example | Mode | Key Timing |
| --- | --- | --- |
| `01_basic_matmul.py` | Sahasra remote | `elapsed_sec = 9.515s` |
| `02_mlp_inference.py` | Sahasra remote | `avg_elapsed_sec = 4.915s` across 3 runs |
| `03_mlp_training.py` | Sahasra remote | `total_elapsed_sec = 207.521s` |
| `04_tiny_transformer.py` | Sahasra remote | `total_elapsed_sec = 90.289s` for 2 epochs |

Notes:

- these runs were billed successfully and executed on `worker-5070ti-demo`
- the current public beta runs on a single RTX 5070 Ti worker, which is also why queue/admission behavior matters under overlap
- the smaller examples are dominated by remote orchestration overhead; the training examples are better representatives of the current remote flow

## Reproducible Comparison Commands

Run these from the repo root after activating your environment:

```bash
python with_sahasra/00_sahasra_jit_decorator.py

python without_sahasra/01_basic_matmul.py
python with_sahasra/01_basic_matmul.py

python without_sahasra/02_mlp_inference.py
python with_sahasra/02_mlp_inference.py

python without_sahasra/03_mlp_training.py
python with_sahasra/03_mlp_training.py --steps-per-execution 8

python without_sahasra/04_tiny_transformer.py
python with_sahasra/04_tiny_transformer.py --steps-per-execution 8

python without_sahasra/05_mnist_cnn.py
python with_sahasra/05_mnist_cnn.py --steps-per-execution 8
python with_sahasra/05_mnist_cnn_inference.py
python with_sahasra/06_pytree_inference.py
python with_sahasra/07_remote_output_tree.py
python with_sahasra/08_checkpoint_roundtrip.py
python with_sahasra/09_repeated_inference.py
python with_sahasra/10_jax_transforms.py
```

For the Sahasra runs, make sure these are set first:

```bash
export SAHASRA_API_URL="https://demo.sahasra.dev"
export SAHASRA_API_BEARER_TOKEN="sk_sahasra_..."
```

## Repo Layout

```text
examples_sahasra/
├── README.md
├── shared/
│   ├── data/
│   │   └── corpus.txt
│   ├── synthetic_mlp.py
│   ├── pure_jax_runtime.py
│   └── tiny_transformer.py
├── without_sahasra/
│   ├── 01_basic_matmul.py
│   ├── 02_mlp_inference.py
│   ├── 03_mlp_training.py
│   └── 04_tiny_transformer.py
└── with_sahasra/
    ├── 00_sahasra_jit_decorator.py
    ├── 01_basic_matmul.py
    ├── 02_mlp_inference.py
    ├── 03_mlp_training.py
    ├── 04_tiny_transformer.py
    ├── 05_mnist_cnn.py
    ├── 05_mnist_cnn_inference.py
    ├── 06_pytree_inference.py
    ├── 07_remote_output_tree.py
    ├── 08_checkpoint_roundtrip.py
    ├── 09_repeated_inference.py
    └── 10_jax_transforms.py
```

## Why Sahasra

The Sahasra examples are designed to show the practical product advantages clearly:

- keep local code and local workflow
- add only a small amount of Sahasra-specific runtime code
- avoid SSH and manual cloud instance setup for many workflows
- use wallet-based billing that is visible from the app and CLI
- keep repeated work warm with `SahasraRuntime`
- pin stable tensors remotely when reuse matters
- move heavier or more VRAM-hungry workloads off the laptop

## Notes

- Your client machine can still be CPU-only. The remote worker is where GPU execution happens.
- Small workloads can still be slower remotely because of orchestration overhead.
- Repeated workloads benefit more from warm sessions and pinned remote tensors.
- Sahasra is designed so you do not hand over your whole codebase or infrastructure story. The runtime sends the execution artifact and the tensors needed for the run, and works best with workers you control.
