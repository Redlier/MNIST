"""项目状态提示工具。"""

from __future__ import annotations

from typing import Final


PENDING_IMPLEMENTATIONS: Final[tuple[str, ...]] = ()


def print_pending_implementations() -> None:
    """打印当前项目的核心实现状态。"""

    if not PENDING_IMPLEMENTATIONS:
        print("项目一核心函数已完成。")
        return
    print("以下函数仍待实现：")
    for function_name in PENDING_IMPLEMENTATIONS:
        print(f"  - {function_name}")


if __name__ == "__main__":
    print_pending_implementations()
