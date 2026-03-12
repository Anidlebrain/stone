# -*- coding: utf-8 -*-
"""
Search best strategy against top10k.npz
Rule: Round 4
- If you lose pile n by <= 5 coins, then on pile n+1 your coins are multiplied by 1.5
- Trigger can happen multiple times across piles

Input:
    top10k.npz
contains:
    strategies: shape (N, 10), int16
    wins: shape (N,), int32   # not used here

Goal:
    Search one strategy x (10 ints, sum=100, each >=0)
    maximizing wins against all strategies in top10k.

Method:
    random init + hill climbing + multiple restarts
"""

import time
import math
import argparse
import numpy as np
import judge

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

BASE_TOTAL = 100
K = 10

# =========================================================
# Core match logic
# =========================================================

if NUMBA_OK:
    @njit(cache=True)
    def match_result(a, b):
        """
        Return 1 if a wins, -1 if b wins, 0 if tie.
        a, b: int16[10] - representing the 10 sandpile values for each player.
        """

        return judge.calculate_match_result(a, b)

    @njit(cache=True)
    def evaluate_against_pool(x, pool):
        """
        x: shape (10,)
        pool: shape (N,10)
        return:
            wins, ties, losses
        """
        wins = 0
        ties = 0
        losses = 0
        for i in range(pool.shape[0]):
            r = match_result(x, pool[i])
            if r == 1:
                wins += 1
            elif r == 0:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses

else:
    def match_result(a, b):
        """
        Return 1 if a wins, -1 if b wins, 0 if tie.
        a, b: int16[10] - representing the 10 sandpile values for each player.
        """

        return judge.calculate_match_result(a, b)

    def evaluate_against_pool(x, pool):
        wins = 0
        ties = 0
        losses = 0
        for i in range(pool.shape[0]):
            r = match_result(x, pool[i])
            if r == 1:
                wins += 1
            elif r == 0:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses


# =========================================================
# Strategy generation / mutation
# =========================================================
def round9_target_sum(x):
    """
    第九轮目标总金币：
    初始100 + 所有被放弃沙堆的奖励
    若第 i 个沙堆(1-based)为0，则奖励 2*i
    """
    bonus = 0
    for i in range(K):
        if int(x[i]) == 0:
            bonus += 2 * (i + 1)
    return BASE_TOTAL + bonus


def is_valid_round9(x):
    x = np.asarray(x)
    if x.shape[0] != K:
        return False
    if np.any(x < 0):
        return False
    return int(x.sum()) == round9_target_sum(x)


def random_strategy_no_rule(rng):
    base = np.ones(K, dtype=np.int16)
    remain = BASE_TOTAL - K
    p = rng.dirichlet(np.ones(K))
    extra = rng.multinomial(remain, p).astype(np.int16)
    return base + extra



def get_abandon_weights(mode: str) -> np.ndarray:
    """
    返回 10 个沙堆被放弃时的采样权重
    mode:
        - low: 更偏向放弃低价值堆
        - high: 更偏向放弃高价值堆
        - mid: 更偏向放弃中间价值堆
        - uniform: 均匀随机
    """
    if mode == "low":
        w = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float64)
    elif mode == "high":
        w = np.array([1, 1, 1, 1, 1, 1, 1, 1, 9, 10], dtype=np.float64)
    elif mode == "mid":
        # 中间更高，两头更低
        w = np.array([2, 4, 6, 8, 10, 10, 8, 6, 4, 2], dtype=np.float64)
    elif mode == "uniform":
        w = np.ones(K, dtype=np.float64)
    else:
        raise ValueError(f"unknown abandon mode: {mode}")

    w /= w.sum()
    return w


def sample_abandoned(rng: np.random.Generator, cnt_range: tuple[int, int]) -> np.ndarray:
    """
    通用弃堆采样：
    cnt_range = (lo, hi)，表示弃堆数量在 [lo, hi] 之间（闭区间）
    风格混合：
        - 35% 低价值优先
        - 35% 高价值优先
        - 15% 中间优先
        - 15% 均匀随机
    """
    lo, hi = cnt_range
    cnt = int(rng.integers(lo, hi + 1))

    p = rng.random()
    if p < 0.20:
        mode = "low"
    elif p < 0.50:
        mode = "high"
    elif p < 0.70:
        mode = "mid"
    else:
        mode = "uniform"

    weights = get_abandon_weights(mode)
    idx = rng.choice(K, size=cnt, replace=False, p=weights)

    abandoned = np.zeros(K, dtype=bool)
    abandoned[idx] = True
    return abandoned


def sample_abandoned_light(rng: np.random.Generator) -> np.ndarray:
    """
    轻度使用规则：
    - 放弃 1~3 个沙堆
    - 弃堆风格混合
    """
    return sample_abandoned(rng, (1, 3))


def sample_abandoned_heavy(rng: np.random.Generator) -> np.ndarray:
    """
    重度使用规则：
    - 放弃 4~7 个沙堆
    - 弃堆风格混合
    """
    return sample_abandoned(rng, (4, 7))


def build_round9_from_abandoned(rng: np.random.Generator, abandoned: np.ndarray) -> np.ndarray:
    """
    给定放弃位置，构造一个第九轮合法策略
    """
    total = BASE_TOTAL + sum(2 * (i + 1) for i in range(K) if abandoned[i])

    v = np.zeros(K, dtype=np.int16)
    active_idx = np.flatnonzero(~abandoned)

    if len(active_idx) == 1:
        v[active_idx[0]] = total
        return v

    p = rng.dirichlet(np.ones(len(active_idx)))
    alloc = rng.multinomial(total, p).astype(np.int16)
    v[active_idx] = alloc
    return v


def random_strategy_light_rule(rng: np.random.Generator) -> np.ndarray:
    abandoned = sample_abandoned_light(rng)
    return build_round9_from_abandoned(rng, abandoned)


def random_strategy_heavy_rule(rng: np.random.Generator) -> np.ndarray:
    abandoned = sample_abandoned_heavy(rng)
    return build_round9_from_abandoned(rng, abandoned)



def build_round9_from_abandoned(rng, abandoned):
    total = BASE_TOTAL + sum(2 * (i + 1) for i in range(K) if abandoned[i])

    x = np.zeros(K, dtype=np.int16)
    active = np.flatnonzero(~abandoned)

    if len(active) == 1:
        x[active[0]] = total
        return x

    p = rng.dirichlet(np.ones(len(active)))
    alloc = rng.multinomial(total, p).astype(np.int16)
    x[active] = alloc
    return x


def random_strategy_light_rule(rng):
    return build_round9_from_abandoned(rng, sample_abandoned_light(rng))


def random_strategy_heavy_rule(rng):
    return build_round9_from_abandoned(rng, sample_abandoned_heavy(rng))


def random_strategy(rng):
    """
    三类混合：
    - 30% 不用规则
    - 40% 轻度用规则
    - 30% 重度用规则
    """
    p = rng.random()
    if p < 0.30:
        return random_strategy_no_rule(rng)
    elif p < 0.70:
        return random_strategy_light_rule(rng)
    else:
        return random_strategy_heavy_rule(rng)

MIN_ACTIVE = 2

def repair_round9(x, rng=None):
    """
    把任意整数向量修复成第九轮合法策略：
    - 所有位置 >= 0
    - 总和 = 100 + 2*所有零位价值之和
    """

    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=np.int16).copy()
    x[x < 0] = 0

    # 修复 1/2 → 3
    small = (x > 0) & (x < MIN_ACTIVE)
    x[small] = MIN_ACTIVE

    for _ in range(30):
        target = round9_target_sum(x)
        cur = int(x.sum())

        if cur == target:
            return x

        active = np.flatnonzero(x >= MIN_ACTIVE)

        if cur < target:
            if len(active) == 0:
                j = int(rng.integers(0, K))
                x[j] = target
                return x
            j = int(active[rng.integers(0, len(active))])
            x[j] += (target - cur)

        else:
            need = cur - target
            order = active.copy()
            rng.shuffle(order)

            for j in order:
                # 不能降到 <3
                take = min(int(x[j] - MIN_ACTIVE), need)
                if take <= 0:
                    continue
                x[j] -= take
                need -= take
                if need == 0:
                    break

    return x


def mutate_strategy(x, rng, max_step=8, moves=3):
    """
    第九轮邻域扰动：
    混合三类动作
    1. 非零堆之间搬金币
    2. 把某个非零堆直接变成0（主动放弃）
    3. 把某个0堆重新启用（取消放弃）
    最后统一 repair_round9
    """
    y = np.asarray(x, dtype=np.int16).copy()

    m = int(rng.integers(1, moves + 1))
    for _ in range(m):
        action = int(rng.integers(0, 3))

        # 动作1：普通搬金币
        if action == 0:
            pos = np.flatnonzero(y > 0)
            if len(pos) == 0:
                continue

            i = int(pos[rng.integers(0, len(pos))])
            j = int(rng.integers(0, K))
            while j == i:
                j = int(rng.integers(0, K))

            available = int(y[i])
            if available <= 0:
                continue

            step = int(rng.integers(1, max_step + 1))
            step = min(step, available)

            y[i] -= step
            y[j] += step

        # 动作2：主动放弃一个非零堆
        elif action == 1:
            pos = np.flatnonzero(y > 0)
            if len(pos) == 0:
                continue

            i = int(pos[rng.integers(0, len(pos))])
            y[i] = 0

        # 动作3：重新启用一个0堆
        else:
            zeros = np.flatnonzero(y == 0)
            pos = np.flatnonzero(y > 0)
            if len(zeros) == 0 or len(pos) == 0:
                continue

            i = int(zeros[rng.integers(0, len(zeros))])
            donor = int(pos[rng.integers(0, len(pos))])
            step = min(int(y[donor]), int(rng.integers(1, max_step + 1)))
            if step <= 0:
                continue

            y[donor] -= step
            y[i] += step

    y = repair_round9(y, rng=rng)
    return y


def jitter_strategy(base, max_delta=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    base = np.asarray(base, dtype=np.int16)
    noise = rng.integers(-max_delta, max_delta + 1, size=base.shape)
    new_x = base + noise
    new_x = repair_round9(new_x, rng=rng)
    return new_x


def structured_seeds(max_delta=3, keep_original=True, rng=None):
    """
    返回种子策略：
    1. 保留原始策略（可选）
    2. 对每条原始策略随机生成 n 条类似策略

    参数
    ----
    n : int
        每条原始策略额外生成多少条
    max_delta : int
        每个位置允许的最大波动幅度
    keep_original : bool
        是否保留 judge.strategies 中原始策略
    rng : np.random.Generator or None
        随机数生成器，便于复现
    """
    n = 100
    if rng is None:
        rng = np.random.default_rng()

    seeds = []

    for s in judge.strategies:
        base = repair_round9(s, rng=rng)

        if keep_original:
            seeds.append(base.copy())

        for _ in range(n):
            new_s = jitter_strategy(base, max_delta=max_delta, rng=rng)
            seeds.append(new_s)

    return seeds

# =========================================================
# Search
# =========================================================


def score_tuple(w, t, l):
    """
    Primary: wins
    Secondary: ties
    """
    return (w, t, -l)


def local_search(pool, x0, rng, iters=3000, neighborhood=40, max_step=8, verbose=False):
    """
    Hill climbing:
    - Around current best x
    - Sample 'neighborhood' mutations
    - Take the best improving move
    - If none improves, do a random shake
    """
    x = x0.copy()
    w, t, l = evaluate_against_pool(x, pool)
    best_local = x.copy()
    best_w, best_t, best_l = w, t, l

    stall = 0

    for it in range(1, iters + 1):
        improved = False
        cand_best = None
        cand_score = None

        for _ in range(neighborhood):
            y = mutate_strategy(x, rng, max_step=max_step, moves=3)
            wy, ty, ly = evaluate_against_pool(y, pool)
            sc = score_tuple(wy, ty, ly)

            if cand_score is None or sc > cand_score:
                cand_score = sc
                cand_best = (y, wy, ty, ly)

        if cand_score > score_tuple(w, t, l):
            x, w, t, l = cand_best
            improved = True
            stall = 0

            if score_tuple(w, t, l) > score_tuple(best_w, best_t, best_l):
                best_local = x.copy()
                best_w, best_t, best_l = w, t, l
        else:
            stall += 1

        # no improvement for a while -> shake current point
        if not improved:
            x = mutate_strategy(
                best_local, rng, max_step=max_step * 2, moves=6)
            w, t, l = evaluate_against_pool(x, pool)

        if verbose and (it % 200 == 0):
            print(
                f"    iter={it:5d} | local_best wins={best_w} ties={best_t} losses={best_l} | stall={stall}")

    return best_local, best_w, best_t, best_l


def global_search(
    pool,
    restarts=30,
    iters=3000,
    neighborhood=40,
    max_step=8,
    seed=1,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    global_best = None
    global_w = -1
    global_t = -1
    global_l = 10**9

    seeds = structured_seeds(rng=rng)
    t0 = time.time()

    for r in range(1, restarts + 1):
        p = rng.random()

        if p < 0.4:
            idx = rng.integers(0, pool.shape[0])
            x0 = repair_round9(pool[idx].copy(), rng=rng)
            x0 = mutate_strategy(x0, rng, max_step=5, moves=2)

        elif p < 0.8:
            x0 = random_strategy(rng)

        else:
            x0 = seeds[rng.integers(0, len(seeds))].copy()

        x, w, t, l = local_search(
            pool=pool,
            x0=x0,
            rng=rng,
            iters=iters,
            neighborhood=neighborhood,
            max_step=max_step,
            verbose=False,
        )

        if score_tuple(w, t, l) > score_tuple(global_w, global_t, global_l):
            global_best = x.copy()
            global_w, global_t, global_l = w, t, l

        if verbose:
            dt = time.time() - t0
            print(
                f"[restart {r:3d}/{restarts}] "
                f"best_this={w}/{t}/{l} | "
                f"global_best={global_w}/{global_t}/{global_l} | "
                f"x={global_best.tolist()} | {dt:.1f}s"
            )

    return global_best, global_w, global_t, global_l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=judge.name + "/top10k.npz")
    parser.add_argument("--restarts", type=int, default=30,
                        help="number of global restarts")
    parser.add_argument("--iters", type=int, default=5000,
                        help="local search iterations per restart")
    parser.add_argument("--neighborhood", type=int, default=40,
                        help="mutations tried per iteration")
    parser.add_argument("--max_step", type=int, default=8,
                        help="max coins moved per mutation")
    parser.add_argument("--seed", type=int, default=202603082229)
    parser.add_argument("--save", type=str,
                        default=judge.name + "/best_vs_top10k.txt")
    parser.add_argument("--min_value", type=int, default=2,
                        help="minimum coins for each pile")
    args = parser.parse_args()
    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Loading {args.input} ...")
    data = np.load(args.input)
    pool = data["strategies"].astype(np.int16)
    # print(pool)
    # seeds = structured_seeds()
    # seeds = np.array(seeds, dtype=np.int16)
    # print(seeds)
    # pool = np.vstack([pool, seeds])
    # print(pool)
    print(f"Pool size = {pool.shape[0]:,}")

    best_x, best_w, best_t, best_l = global_search(
        pool=pool,
        restarts=args.restarts,
        iters=args.iters,
        neighborhood=args.neighborhood,
        max_step=args.max_step,
        seed=args.seed,
        verbose=True,
    )

    print("\n=== FINAL BEST ===")
    print("strategy =", best_x.tolist())
    print(f"wins={best_w}, ties={best_t}, losses={best_l}")
    print(f"win_rate={best_w / pool.shape[0]:.4f}")

    with open(args.save, "a", encoding="utf8") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} wins={best_w} ties={best_t} losses={best_l} win_rate={best_w / pool.shape[0]:.6f}\n"
        )
        f.write("strategy: " + " ".join(map(str, best_x.tolist())) + "\n")

    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
