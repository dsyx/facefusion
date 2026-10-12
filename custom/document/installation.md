# 安装指南

## 官方 Conda 安装

参考官方 [Installation](../../README.md#installation)。

## 免 Conda 安装（Windows）

前置依赖：

- Windows 10/11 (64-bit)
- NVIDIA GPU with a CUDA 12 compatible driver
- [Git](https://git-scm.com/)
- [FFmpeg](https://ffmpeg.org/)
- [Python 3.12](https://www.python.org/)

### 使用 uv（推荐）

[`uv`](https://docs.astral.sh/uv/) 具备极速依赖解析和虚拟环境管理能力，支持免激活环境直接运行。

```powershell
# Create a virtual environment (--seed ensures pip is included for install.py)
uv venv --python 3.12 --seed

# Install CUDA 12 and cuDNN runtime libraries
uv pip install `
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 `
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 `
  nvidia-cusparse-cu12 nvidia-nvjitlink-cu12

# Install FaceFusion dependencies
uv run python install.py cuda@12 --skip-conda

# Run FaceFusion
uv run python facefusion.py run
```

### 使用传统 venv

使用 Python 标准库自带的虚拟环境管理器。

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the environment
.venv\Scripts\activate

# Install CUDA 12 and cuDNN runtime libraries
python -m pip install `
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 `
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 `
  nvidia-cusparse-cu12 nvidia-nvjitlink-cu12

# Install FaceFusion dependencies
python install.py cuda@12 --skip-conda

# Run FaceFusion
python facefusion.py run
```

