"""下载并校验 MNIST 数据集。"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "mnist.npz"

# 第一个是 TensorFlow 官方公开文件，第二个是 Hugging Face 的国内镜像域名。
# 镜像偶尔会同步延迟，所以脚本会按顺序尝试所有地址。
DOWNLOAD_URLS: tuple[str, ...] = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    "https://hf-mirror.com/datasets/ylecun/mnist/resolve/main/mnist.npz",
)


def _is_valid_mnist_file(file_path: Path) -> bool:
    """检查文件是否包含 MNIST 所需的四个数组。"""

    if not file_path.exists() or file_path.stat().st_size < 1024:
        return False
    try:
        with np.load(file_path, allow_pickle=False) as data:
            required_keys = {"x_train", "y_train", "x_test", "y_test"}
            return required_keys.issubset(set(data.files))
    except (OSError, ValueError, KeyError):
        return False


def download_mnist(
    output_path: Path = DEFAULT_OUTPUT,
    urls: Iterable[str] = DOWNLOAD_URLS,
    timeout_seconds: int = 30,
) -> Path:
    """从多个来源尝试下载 MNIST，并返回本地文件路径。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_valid_mnist_file(output_path):
        print(f"MNIST 已存在且通过检查：{output_path}")
        return output_path

    errors: list[str] = []
    for url in urls:
        temporary_path = output_path.with_suffix(output_path.suffix + ".part")
        try:
            print(f"正在尝试下载：{url}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (MNIST teaching scaffold)"},
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                with temporary_path.open("wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
            if not _is_valid_mnist_file(temporary_path):
                raise ValueError("下载内容不是可识别的 MNIST npz 文件")
            temporary_path.replace(output_path)
            print(f"下载完成：{output_path}")
            return output_path
        except Exception as error:  # 网络异常类型很多，统一转成易懂信息
            errors.append(f"{url} -> {type(error).__name__}: {error}")
            if temporary_path.exists():
                temporary_path.unlink()

    joined_errors = "\n".join(f"  - {item}" for item in errors)
    raise RuntimeError(
        "MNIST 下载失败。请检查网络，或手动把 mnist.npz 放入 data/ 目录。\n"
        f"已尝试来源：\n{joined_errors}"
    )


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="下载 NumPy MLP 项目的 MNIST 数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出文件路径，默认是项目 data/mnist.npz",
    )
    args = parser.parse_args()
    try:
        download_mnist(args.output)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
