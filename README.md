# Sahasra Examples

This repo is a public companion repo for **Sahasra**.

It is organized to make one idea very easy to understand:

**the same kind of JAX code can be run locally or through Sahasra with only small changes.**

So the repo is split into two parallel folders:

- [without_sahasra](./without_sahasra): plain local JAX examples
- [with_sahasra](./with_sahasra): the same style of examples adapted to run on Sahasra

The examples are ordered from basic to more demanding:

1. `01_basic_matmul.py`
2. `02_mlp_inference.py`
3. `03_mlp_training.py`
4. `04_tiny_transformer.py`

## What This Repo Is For

This repo is for people who want to answer questions like:

- What changes when I move a local JAX example to Sahasra?
- How small can the code diff be?
- How do warm sessions and `SahasraRuntime` fit into repeated work?
- How does remote training look compared to local training?

It is not trying to hide the difference between local and remote execution.
It is trying to make that difference understandable and practical.

## Install

### Local-only JAX examples

If you only want to run the `without_sahasra/` examples:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "jax>=0.4.35" "numpy>=1.26"
```

### Sahasra examples from TestPyPI

If you want to run the `with_sahasra/` examples:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://pypi.org/simple \
  "httpx>=0.27,<1" \
  "pydantic>=2.9,<3" \
  "flatbuffers>=24.3.25" \
  "jax>=0.4.35"

python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  "sahasra-core==0.1.0"

python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  "sahasra==0.1.0"
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

## Run The Examples

### Local

```bash
python without_sahasra/01_basic_matmul.py
python without_sahasra/02_mlp_inference.py
python without_sahasra/03_mlp_training.py
python without_sahasra/04_tiny_transformer.py --epochs 2
```

### Sahasra

```bash
python with_sahasra/01_basic_matmul.py
python with_sahasra/02_mlp_inference.py
python with_sahasra/03_mlp_training.py --epochs 2 --steps-per-execution 2
python with_sahasra/04_tiny_transformer.py --epochs 2 --steps-per-execution 2
```

## Repo Layout

```text
examples_sahasra/
├── README.md
├── shared/
│   ├── data/
│   │   └── corpus.txt
│   ├── synthetic_mlp.py
│   └── tiny_transformer.py
├── without_sahasra/
│   ├── 01_basic_matmul.py
│   ├── 02_mlp_inference.py
│   ├── 03_mlp_training.py
│   └── 04_tiny_transformer.py
└── with_sahasra/
    ├── 01_basic_matmul.py
    ├── 02_mlp_inference.py
    ├── 03_mlp_training.py
    └── 04_tiny_transformer.py
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
