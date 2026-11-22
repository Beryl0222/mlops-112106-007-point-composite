"""命令行入口：按时间线回放一个交换场景并打印结构化结果。

用法::

    python3 run_session.py samples/scene.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alpha_compose.load import load_scene
from alpha_compose.timeline import play


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="点投影深度合成会话回放")
    parser.add_argument("scene", type=Path, help="交换场景 JSON 路径")
    args = parser.parse_args()
    scene = load_scene(args.scene)
    result = play(scene)
    result.pop("session")
    print(json.dumps(result, ensure_ascii=False, indent=2, cls=_Encoder))


if __name__ == "__main__":
    main()
