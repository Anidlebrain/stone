import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# =========================
# Round 5 Rule Search Script
# =========================
# Rule:
# For each pile k, your effective allocation is:
#   max(0, original[k] - losses_before_k)
# where losses_before_k is how many earlier piles you have lost.
# Whoever has larger total pile value wins the match.
# Equal effective allocation on a pile -> nobody gets that pile.
# Ranking is by number of opponents beaten.
#
# Workflow in this script:
# 1) Generate 1,000,000 unique strategies.
# 2) Run 1000 Monte Carlo rounds to estimate strength.
# 3) Keep top 10,000 and save them.
# 4) Build 8 seed families:
#    - 稳健型
#    - 前压型
#    - 前中均衡型
#    - 卡位型
#    - 反制后置型
#    - 放弃10
#    - 放弃9/10
#    - 前面模拟的1W名（直接从 top10k 中抽起点）
# 5) Random-restart local search from these seeds and output best result.
#
# Notes:
# - The exact “1,000,000 x 1,000 rounds x full round-robin” is computationally huge.
# - So this script uses staged Monte Carlo estimation, which is much more practical.
# - You can increase/decrease sample sizes with the parameters below.

TOTAL_COINS = 100
NUM_PILES = 10
PILE_VALUES = np.arange(1, NUM_PILES + 1, dtype=np.int16)

# ---------- Stage 1 params ----------
POP_SIZE = 1_000_000
TOP_K = 10_000
MC_ROUNDS = 1000
# In stage 1, each round does a random full matching: every strategy is paired once.
# Score only counts wins. Ties and losses are both 0, matching your ranking rule.
OPPONENTS_PER_ROUND = 1
BATCH_SIZE = 20_000            # process candidate strategies by batches
SAVE_TOP_PATH = "top10k_round5.csv"
SAVE_BEST_PATH = "best_round5_search.txt"
SEED = 42

# ---------- Stage 2 params ----------
SEARCH_RESTARTS_PER_FAMILY = 40
SEARCH_STEPS = 2500
SEARCH_MUTATIONS_PER_STEP = 12
# number of elite opponents used in local search evaluation
EVAL_POOL_SIZE = 2500
# evaluate final best against all top10k if top10k size is 10k
FINAL_FULL_EVAL_SIZE = 10_000


# =========================
# Core game logic
# =========================
def duel_round5(a: np.ndarray, b: np.ndarray) -> int:
    """
    Return:
      1 if a beats b
      0 if tie
     -1 if a loses to b
    """
    loss_a = 0
    loss_b = 0
    score_a = 0
    score_b = 0

    for i in range(NUM_PILES):
        ea = a[i] - loss_a
        eb = b[i] - loss_b
        if ea < 0:
            ea = 0
        if eb < 0:
            eb = 0

        if ea > eb:
            score_a += i + 1
            loss_b += 1
        elif ea < eb:
            score_b += i + 1
            loss_a += 1
        # equal -> nobody scores, no new loss

    if score_a > score_b:
        return 1
    if score_a < score_b:
        return -1
    return 0


def eval_vs_pool(strategy: np.ndarray, pool: np.ndarray) -> Tuple[int, int, int, float]:
    wins = 0
    ties = 0
    losses = 0
    for opp in pool:
        r = duel_round5(strategy, opp)
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            ties += 1
    wr = wins / len(pool) if len(pool) else 0.0
    return wins, ties, losses, wr


# =========================
# Strategy generation
# =========================
def random_composition_100_10(rng: np.random.Generator) -> np.ndarray:
    """Uniform random composition of 100 into 10 nonnegative integers."""
    cuts = rng.choice(TOTAL_COINS + NUM_PILES - 1,
                      size=NUM_PILES - 1, replace=False)
    cuts.sort()
    arr = np.empty(NUM_PILES, dtype=np.int16)
    prev = -1
    for i, c in enumerate(cuts):
        arr[i] = c - prev - 1
        prev = c
    arr[-1] = (TOTAL_COINS + NUM_PILES - 2) - prev
    return arr


def generate_unique_population(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seen = set()
    out = np.empty((n, NUM_PILES), dtype=np.int16)
    filled = 0
    while filled < n:
        x = random_composition_100_10(rng)
        key = tuple(int(v) for v in x)
        if key in seen:
            continue
        seen.add(key)
        out[filled] = x
        filled += 1
        if filled % 100_000 == 0:
            print(f"[gen] generated {filled}/{n}")
    return out


# =========================
# Seed families
# =========================
def normalize_to_100(x: List[int]) -> np.ndarray:
    arr = np.array(x, dtype=np.int16)
    s = int(arr.sum())
    if s == TOTAL_COINS:
        return arr
    scaled = np.floor(arr * TOTAL_COINS / s).astype(np.int16)
    diff = TOTAL_COINS - int(scaled.sum())
    order = np.argsort(-(arr * TOTAL_COINS / s - scaled))
    for i in range(diff):
        scaled[order[i % NUM_PILES]] += 1
    return scaled


def seed_stable(rng: np.random.Generator) -> np.ndarray:
    base = [8, 8, 9, 9, 10, 10, 11, 11, 12, 12]
    arr = normalize_to_100(base)
    return jitter_keep_sum(arr, rng, steps=15, max_move=2)


def seed_front_press(rng: np.random.Generator) -> np.ndarray:
    base = [16, 15, 14, 13, 11, 9, 8, 6, 4, 4]
    arr = normalize_to_100(base)
    return jitter_keep_sum(arr, rng, steps=18, max_move=3)


def seed_front_mid_balanced(rng: np.random.Generator) -> np.ndarray:
    base = [12, 12, 12, 12, 11, 10, 10, 8, 7, 6]
    arr = normalize_to_100(base)
    return jitter_keep_sum(arr, rng, steps=18, max_move=2)


def seed_card_point(rng: np.random.Generator) -> np.ndarray:
    base = [14, 7, 14, 7, 13, 8, 12, 8, 9, 8]
    arr = normalize_to_100(base)
    return jitter_keep_sum(arr, rng, steps=20, max_move=3)


def seed_counter_backload(rng: np.random.Generator) -> np.ndarray:
    base = [14, 13, 12, 11, 10, 10, 10, 8, 7, 5]
    arr = normalize_to_100(base)
    return jitter_keep_sum(arr, rng, steps=18, max_move=2)


def seed_giveup_10(rng: np.random.Generator) -> np.ndarray:
    base = [10, 10, 11, 11, 12, 12, 12, 12, 10, 0]
    arr = normalize_to_100(base)
    return jitter_keep_sum(arr, rng, steps=18, max_move=3)


def seed_giveup_9_10(rng: np.random.Generator) -> np.ndarray:
    base = [13, 13, 13, 12, 12, 12, 11, 14, 0, 0]
    arr = normalize_to_100(base)
    return jitter_keep_sum(arr, rng, steps=18, max_move=3)


def seed_from_top10k(rng: np.random.Generator, top10k: np.ndarray) -> np.ndarray:
    idx = int(rng.integers(0, len(top10k)))
    arr = np.array(top10k[idx], dtype=np.int16)
    return jitter_keep_sum(arr, rng, steps=12, max_move=2)


# =========================
# Mutation / neighborhood
# =========================
def jitter_keep_sum(arr: np.ndarray, rng: np.random.Generator, steps: int = 10, max_move: int = 2) -> np.ndarray:
    x = arr.copy()
    for _ in range(steps):
        i, j = rng.choice(NUM_PILES, size=2, replace=False)
        if x[j] <= 0:
            continue
        mv = int(rng.integers(1, max_move + 1))
        mv = min(mv, int(x[j]))
        x[i] += mv
        x[j] -= mv
    return x


def mutate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    mode = int(rng.integers(0, 5))

    if mode == 0:
        # simple transfer
        i, j = rng.choice(NUM_PILES, size=2, replace=False)
        if y[j] > 0:
            mv = min(int(rng.integers(1, 5)), int(y[j]))
            y[i] += mv
            y[j] -= mv
    elif mode == 1:
        # stronger shift to earlier piles
        i = int(rng.integers(0, 5))
        j = int(rng.integers(5, 10))
        if y[j] > 0:
            mv = min(int(rng.integers(1, 6)), int(y[j]))
            y[i] += mv
            y[j] -= mv
    elif mode == 2:
        # stronger shift to mid piles
        i = int(rng.integers(2, 7))
        j = int(rng.integers(0, 10))
        if i != j and y[j] > 0:
            mv = min(int(rng.integers(1, 5)), int(y[j]))
            y[i] += mv
            y[j] -= mv
    elif mode == 3:
        # smooth local profile
        a = int(rng.integers(0, NUM_PILES - 1))
        b = a + 1
        c = int(rng.integers(0, NUM_PILES))
        if c not in (a, b) and y[c] >= 2:
            y[a] += 1
            y[b] += 1
            y[c] -= 2
    else:
        # random multi-jitter
        y = jitter_keep_sum(y, rng, steps=4, max_move=3)

    return y


# =========================
# Stage 1: estimate top 10k
# =========================
@dataclass
class Stage1Result:
    population: np.ndarray
    scores: np.ndarray
    top_idx: np.ndarray
    top_pop: np.ndarray
    top_scores: np.ndarray


def estimate_population_strength(pop: np.ndarray, rounds: int, opponents_per_round: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(pop)
    scores = np.zeros(n, dtype=np.int32)

    if n % 2 != 0:
        raise ValueError(
            "Population size must be even for full random matching per round.")

    # Each round: shuffle everyone, then pair (0,1), (2,3), ...
    # Every strategy appears exactly once per round.
    # Score rule: only wins count. Tie = 0, Loss = 0.
    order = np.arange(n, dtype=np.int32)
    for rd in range(rounds):
        rng.shuffle(order)
        if (rd + 1) % 10 == 0 or rd == 0:
            print(f"[mc] round {rd + 1}/{rounds}")

        for i in range(0, n, 2):
            ia = int(order[i])
            ib = int(order[i + 1])
            r = duel_round5(pop[ia], pop[ib])
            if r > 0:
                scores[ia] += 1
            elif r < 0:
                scores[ib] += 1
            # tie -> nobody gets a win

    return scores.astype(np.float32)


def build_top10k(pop: np.ndarray, scores: np.ndarray, top_k: int) -> Stage1Result:
    idx = np.argpartition(scores, -top_k)[-top_k:]
    idx = idx[np.argsort(-scores[idx])]
    return Stage1Result(
        population=pop,
        scores=scores,
        top_idx=idx,
        top_pop=pop[idx].copy(),
        top_scores=scores[idx].copy(),
    )


def save_top10k_csv(path: str, top_pop: np.ndarray, top_scores: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "wins", *[f"p{i}" for i in range(1, 11)]])
        for rank, (s, sc) in enumerate(zip(top_pop, top_scores), start=1):
            writer.writerow([rank, int(sc), *map(int, s)])
    print(f"[save] top10k saved to {path}")


# =========================
# Stage 2: local search on elites
# =========================
def choose_eval_pool(top10k: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    size = min(size, len(top10k))
    idx = rng.choice(len(top10k), size=size, replace=False)
    return top10k[idx]


def fast_score(strategy: np.ndarray, pool: np.ndarray) -> Tuple[int, int, int, float]:
    return eval_vs_pool(strategy, pool)


def hill_climb(start: np.ndarray, pool: np.ndarray, rng: np.random.Generator, steps: int) -> Tuple[np.ndarray, Tuple[int, int, int, float]]:
    cur = start.copy()
    cur_stat = fast_score(cur, pool)

    for step in range(steps):
        best_local = cur
        best_stat = cur_stat
        for _ in range(SEARCH_MUTATIONS_PER_STEP):
            cand = mutate(cur, rng)
            stat = fast_score(cand, pool)
            # ranking rule: only wins matter; tie and loss are equivalent
            key_c = (stat[0],)
            key_b = (best_stat[0],)
            if key_c > key_b:
                best_local = cand
                best_stat = stat

        if best_local is cur:
            # small random kick
            cur = jitter_keep_sum(cur, rng, steps=6, max_move=3)
            cur_stat = fast_score(cur, pool)
        else:
            cur = best_local
            cur_stat = best_stat

        if (step + 1) % 500 == 0:
            print(
                f"    [climb] step {step + 1}/{steps}, wins={cur_stat[0]}, ties={cur_stat[1]}, losses={cur_stat[2]}")

    return cur, cur_stat


def run_seed_family_search(top10k: np.ndarray, seed: int) -> Tuple[np.ndarray, Tuple[int, int, int, float], str]:
    rng = np.random.default_rng(seed)
    family_builders = [
        # ("稳健型", lambda: seed_stable(rng)),
        # ("前压型", lambda: seed_front_press(rng)),
        # ("前中均衡型", lambda: seed_front_mid_balanced(rng)),
        # ("卡位型", lambda: seed_card_point(rng)),
        # ("反制后置型", lambda: seed_counter_backload(rng)),
        # ("放弃10", lambda: seed_giveup_10(rng)),
        # ("放弃9/10", lambda: seed_giveup_9_10(rng)),
        ("前面模拟的1W名", lambda: seed_from_top10k(rng, top10k)),
    ]

    overall_best = None
    overall_best_stat = None
    overall_family = None

    for family_name, builder in family_builders:
        print(f"[family] {family_name}")
        for restart in range(SEARCH_RESTARTS_PER_FAMILY):
            pool = choose_eval_pool(top10k, EVAL_POOL_SIZE, rng)
            start = builder()
            cand, stat = hill_climb(start, pool, rng, SEARCH_STEPS)
            print(
                f"  [restart {restart + 1}/{SEARCH_RESTARTS_PER_FAMILY}] "
                f"wins={stat[0]}, ties={stat[1]}, losses={stat[2]}, wr={stat[3]:.4f}, cand={cand.tolist()}"
            )

            if overall_best is None:
                overall_best = cand.copy()
                overall_best_stat = stat
                overall_family = family_name
            else:
                key_c = (stat[0],)
                key_b = (overall_best_stat[0],)
                if key_c > key_b:
                    overall_best = cand.copy()
                    overall_best_stat = stat
                    overall_family = family_name
                    print(
                        f"  [update] new best from {family_name}: {overall_best.tolist()}")

    return overall_best, overall_best_stat, overall_family


# =========================
# Save final best
# =========================
def save_best(path: str, strategy: np.ndarray, family: str, stat: Tuple[int, int, int, float], final_stat: Tuple[int, int, int, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== BEST ROUND 5 SEARCH RESULT ===\n")
        f.write(f"seed_family = {family}\n")
        f.write(f"strategy = {strategy.tolist()}\n")
        f.write(f"sum = {int(strategy.sum())}\n")
        f.write("\n")
        f.write("=== DURING SEARCH (sampled elite pool) ===\n")
        f.write(f"wins = {stat[0]}\n")
        f.write(f"ties = {stat[1]}\n")
        f.write(f"losses = {stat[2]}\n")
        f.write(f"win_rate = {stat[3]:.6f}\n")
        f.write("\n")
        f.write("=== FINAL EVAL (full top10k if available) ===\n")
        f.write(f"wins = {final_stat[0]}\n")
        f.write(f"ties = {final_stat[1]}\n")
        f.write(f"losses = {final_stat[2]}\n")
        f.write(f"win_rate = {final_stat[3]:.6f}\n")
    print(f"[save] best result saved to {path}")


# =========================
# Main
# =========================
def main() -> None:
    print("[start] generating 1,000,000 unique strategies...")
    pop = generate_unique_population(POP_SIZE, SEED)

    print("[start] stage 1 Monte Carlo scoring (random full matching, wins only)...")
    scores = estimate_population_strength(
        pop=pop,
        rounds=MC_ROUNDS,
        opponents_per_round=OPPONENTS_PER_ROUND,
        seed=SEED + 1,
    )

    print("[start] selecting top10k...")
    s1 = build_top10k(pop, scores, TOP_K)
    save_top10k_csv(SAVE_TOP_PATH, s1.top_pop, s1.top_scores)

    print("[start] stage 2 local search from 8 seed families...")
    best, stat, family = run_seed_family_search(s1.top_pop, seed=SEED + 2)

    print("[start] final evaluation vs top10k...")
    final_pool = s1.top_pop[: min(FINAL_FULL_EVAL_SIZE, len(s1.top_pop))]
    final_stat = eval_vs_pool(best, final_pool)

    print("\n=== FINAL BEST ===")
    print(f"seed_family = {family}")
    print(f"strategy = {best.tolist()}")
    print(f"sum = {int(best.sum())}")
    print(
        f"sample_eval: wins={stat[0]}, ties={stat[1]}, losses={stat[2]}, wr={stat[3]:.4f}")
    print(
        f"final_eval : wins={final_stat[0]}, ties={final_stat[1]}, losses={final_stat[2]}, wr={final_stat[3]:.4f}")

    save_best(SAVE_BEST_PATH, best, family, stat, final_stat)


if __name__ == "__main__":
    main()
