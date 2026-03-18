import numpy as np
from collections import defaultdict
import judge15 as judge

def analyze_npz(path: str):
    data = np.load(path)
    strategies = data["strategies"]
    wins = data["wins"]

    n = strategies.shape[0]

    # 排名：wins 越大排名越靠前
    order = np.argsort(-wins)
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = np.arange(1, n + 1)

    # 按技能分类
    groups = defaultdict(list)

    for i in range(n):
        v = strategies[i]
        skill = f"{int(v[0])}{int(v[1])}{int(v[2])}"
        groups[skill].append(ranks[i])

    print(f"\n=== 分析文件: {path} ===\n")

    for skill in ["000","100","010","001","110","101","011","111"]:
        arr = np.array(groups[skill], dtype=np.int32)

        if len(arr) == 0:
            print(f"{skill}: count=0")
            continue

        print(
            f"{skill}: "
            f"count={len(arr):4d} | "
            f"avg={arr.mean():8.2f} | "
            f"median={np.median(arr):6.1f} | "
            f"best={arr.min():4d} | "
            f"worst={arr.max():4d}"
        )


path = judge.name + '/top100.npz'

if __name__ == "__main__":
    analyze_npz(path)