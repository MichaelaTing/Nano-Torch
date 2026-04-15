# Nano Torch

Nano Torch is a lightweight deep learning framework implementation that includes:
- Automatic differentiation (autograd)
- Core operators and tensor backends
- Neural network modules (MLP / CNN / RNN / LSTM / Transformer)
- Optimizers (SGD / Adam)
- Data loading and common dataset utilities
- C++/CUDA backend extensions (compiled via pybind11)

## Project Structure

```text
Nano Torch/
├── python/needle/          # Core framework implementation
│   ├── autograd.py
│   ├── ops/
│   ├── nn/
│   ├── data/
│   ├── init/
│   ├── optim.py
│   └── backend_ndarray/
├── src/                    # C++/CUDA backend source code
├── apps/                   # Training scripts and model definitions
├── tests/                  # Test suites
├── data/                   # Dataset directory
├── hw1.ipynb ~ hw4_extra.ipynb
├── Makefile
└── CMakeLists.txt
```

## Environment Requirements

A Linux + Conda environment is recommended:
- Python 3.10+ (the project's `Makefile` defaults to `~/anaconda3/envs/llm/bin/python`)
- `cmake` (>= 3.5)
- C/C++ compiler (g++/clang)
- `pybind11`
- `numpy`, `pytest`, `black` (optional)
- For CUDA backend support:
  - NVIDIA driver
  - CUDA Toolkit

## Quick Start

Run the following commands in the project root:

```bash
# 1) Activate environment (example)
conda activate llm

# 2) Install basic dependencies (as needed)
pip install numpy pytest pybind11

# 3) Build the C++ backend
make lib
```

After a successful build, you will typically see files like the following under `python/needle/backend_ndarray/`:
- `ndarray_backend_cpu*.so`
- (optional) `ndarray_backend_cuda*.so`

## Running and Development

### 1) Run Notebooks

Open the homework notebooks in the project root directly:
- `hw1.ipynb`
- `hw2.ipynb`
- `hw3.ipynb`
- `hw4.ipynb`
- `hw4_extra.ipynb`

### 2) Run Tests

Run these commands in the project root (and make sure Python can find `python/`):

```bash
PYTHONPATH=./python pytest tests/hw1 -q
PYTHONPATH=./python pytest tests/hw2 -q
PYTHONPATH=./python pytest tests/hw3 -q
PYTHONPATH=./python pytest tests/hw4 -q
PYTHONPATH=./python pytest tests/hw4_extra -q
```

### 3) Run an Example Script

```bash
PYTHONPATH=./python python apps/simple_ml.py
```

## Backend Switching

The framework chooses a backend via environment variables:
- `NEEDLE_BACKEND=nd`: default, uses the custom NDArray backend
- `NEEDLE_BACKEND=np`: uses the NumPy backend (useful for debugging)

Example:

```bash
NEEDLE_BACKEND=np PYTHONPATH=./python pytest tests/hw1 -q
```
