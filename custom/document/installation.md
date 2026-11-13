# 安装指南

## 官方 Conda 安装

参考官方 [Installation](https://docs.facefusion.io/installation)。

## 免 Conda 安装（Windows）

前置依赖：

- Windows 10/11 (64-bit)
- NVIDIA GPU with a CUDA 12/13 compatible driver
- [Git](https://git-scm.com/)
- [FFmpeg](https://ffmpeg.org/)
- [Python 3.12](https://www.python.org/)

### 使用 uv（推荐）

[`uv`](https://docs.astral.sh/uv/) 具备极速依赖解析和虚拟环境管理能力，支持免激活环境直接运行。

#### CUDA 12（`Driver >= 527.41`）

```powershell
# Create a virtual environment (--seed ensures pip is included for install.py)
uv venv --python 3.12 --seed

# Install CUDA 12, cuDNN and TensorRT runtime libraries
uv pip install `
  "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvjitlink,nvrtc]==12.*" `
  nvidia-cudnn-cu12 `
  tensorrt-cu12-libs==10.*

# Install FaceFusion dependencies
uv run python install.py cuda@12 --skip-conda

# Run FaceFusion
uv run python -X utf8 facefusion.py run
```

#### CUDA 13（`Driver >= 580.x`）

```powershell
# Create a virtual environment (--seed ensures pip is included for install.py)
uv venv --python 3.12 --seed

# Install CUDA 13, cuDNN and TensorRT runtime libraries
uv pip install `
  "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvjitlink,nvrtc]==13.*" `
  nvidia-cudnn-cu13 `
  tensorrt-cu13-libs==10.*

# Install FaceFusion dependencies
uv run python install.py cuda@13 --skip-conda

# Run FaceFusion
uv run python -X utf8 facefusion.py run
```

### 使用传统 venv

使用 Python 标准库自带的虚拟环境管理器。

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the environment
.venv\Scripts\activate

# Install CUDA 12, cuDNN and TensorRT runtime libraries
python -m pip install `
  "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvjitlink,nvrtc]==12.*" `
  nvidia-cudnn-cu12 `
  tensorrt-cu12-libs==10.*

# Install FaceFusion dependencies
python install.py cuda@12 --skip-conda

# Run FaceFusion
python -X utf8 facefusion.py run
```

> 注：如需使用 CUDA 13，请参考 [CUDA 13（`Driver >= 580.x`）](#cuda-13driver--580x) 替换对应依赖包与安装参数。
