"""检查交换资料的形状和索引。"""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    return {
        "资料": path.name,
        "字段": sorted(data),
        "字节": path.stat().st_size,
        "数值环境": np.__version__,
    }


def validate(data: dict) -> None:
    for hits in data["tiles"].values():
            for point, y, x, coverage in hits:
                desc = data["points"][point]
                piece = data["expected_slices"][desc["slice"]]
                y0, y1, x0, x1 = piece["rect"]
                assert y0 <= y < y1 and x0 <= x < x1 and 0 <= coverage <= 1
                assert piece["depth"][0] <= desc["depth"] < piece["depth"][1]


def main() -> None:
    parser = argparse.ArgumentParser(description="检查本地交换资料")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.scene), ensure_ascii=False))


if __name__ == "__main__":
    main()
