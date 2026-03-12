import random
from typing import List, Tuple

VALUES = list(range(1, 11))   # pile values 1..10
TOTAL_COINS = 100
P = 10

# -----------------------------
# Round-2 scoring
# -----------------------------
def first_ge11_index(bids: List[int]) -> int:
    for i, x in enumerate(bids):
        if x >= 11:
            return i
    return -1

def match_score(a: List[int], b: List[int]) -> Tuple[int, int]:
    """Return (score_a, score_b) under round-2 burn rule + Blotto win piles."""
    burn_a = first_ge11_index(a)
    burn_b = first_ge11_index(b)

    sa = sb = 0
    for i, v in enumerate(VALUES):
        if a[i] > b[i]:
            if i != burn_a:
                sa += v
        elif b[i] > a[i]:
            if i != burn_b:
                sb += v
    return sa, sb

def result_vs(a: List[int], b: List[int]) -> int:
    """+1 a win, 0 tie, -1 lose."""
    sa, sb = match_score(a, b)
    if sa > sb:
        return 1
    if sb > sa:
        return -1
    return 0

def wins_ties_losses(me: List[int], opponents: List[List[int]]) -> Tuple[int, int, int]:
    w = t = l = 0
    for o in opponents:
        r = result_vs(me, o)
        if r == 1:
            w += 1
        elif r == 0:
            t += 1
        else:
            l += 1
    return w, t, l

# -----------------------------
# Helpers
# -----------------------------
def trim_to_budget(vec: List[int], budget: int) -> List[int]:
    """
    Ensure sum(vec) <= budget by randomly decrementing positive entries.
    """
    vec = vec[:]
    s = sum(vec)
    while s > budget:
        i = random.randrange(len(vec))
        if vec[i] > 0:
            vec[i] -= 1
            s -= 1
    return vec

def distribute_rest_with_cap(x: List[int], free_idx: List[int], rest: int, cap_piles: set = None, cap_value: int = 3):
    """
    Distribute 'rest' coins into x (free-slot vector), optionally capping certain piles.
    cap_piles: set of pile numbers (1..10) to cap at cap_value-1? (we use >=cap_value to block)
    """
    if cap_piles is None:
        cap_piles = set()

    # loop until all coins placed
    while rest > 0:
        pos = random.randrange(len(free_idx))
        pile = free_idx[pos] + 1
        if pile in cap_piles and x[pos] >= cap_value:
            continue
        x[pos] += 1
        rest -= 1

def rand_partition(total: int, k: int, min_each: int = 0) -> List[int]:
    """Random integer partition: k parts, each >= min_each, sum=total."""
    assert total >= k * min_each
    remaining = total - k * min_each
    # stars and bars
    cuts = sorted(random.sample(range(remaining + k - 1), k - 1))
    parts = []
    prev = -1
    for c in cuts + [remaining + k - 1]:
        parts.append(c - prev - 1)
        prev = c
    return [p + min_each for p in parts]

def sprinkle(maxv: int = 3) -> int:
    return random.randint(0, maxv)

# -----------------------------
# Bone (通用骨架) enforcement
# Only two allowed:
#   A: [11, *, *, ..., *]   (pile1=11)
#   B: [1, 11, *, ..., *]   (pile1=1, pile2=11)
# -----------------------------
def make_bone() -> Tuple[List[int], List[int], int]:
    """
    Returns (prefix, free_indices, free_sum)
    prefix: length 10 array with fixed values at fixed positions else None marker (-1)
    free_indices: list of indices you can fill
    free_sum: coins to allocate to free_indices
    """
    use_A = (random.random() < 0.5)
    x = [-1] * P
    if use_A:
        x[0] = 11
        free_indices = list(range(1, P))
        free_sum = TOTAL_COINS - 11  # 89
    else:
        x[0] = 1
        x[1] = 11
        free_indices = list(range(2, P))
        free_sum = TOTAL_COINS - 12  # 88
    return x, free_indices, free_sum

def fill_free(prefix: List[int], free_indices: List[int], amounts: List[int]) -> List[int]:
    x = prefix[:]
    assert len(free_indices) == len(amounts)
    for idx, amt in zip(free_indices, amounts):
        x[idx] = amt
    assert all(v >= 0 for v in x)
    if sum(x) != TOTAL_COINS:
        raise ValueError(f"sum!=100, sum={sum(x)}, x={x}")
    assert sum(x) == TOTAL_COINS
    # sanity: bone validity
    assert (x[0] == 11) or (x[0] == 1 and x[1] == 11)
    return x

# -----------------------------
# Opponent generators (6 types)
# All except type3 obey bone strictly.
# -----------------------------
def gen_type1() -> List[int]:
    """(1) bone + main attack 8/9/10; others small or 0."""
    prefix, free_idx, free_sum = make_bone()
    x = [0] * len(free_idx)

    # sprinkle with budget safety
    for pos_i, real_i in enumerate(free_idx):
        pile = real_i + 1
        if pile not in (8, 9, 10) and random.random() < 0.5:
            x[pos_i] = sprinkle(3)

    # ensure we don't exceed free_sum
    x = trim_to_budget(x, free_sum)
    remaining = free_sum - sum(x)

    # allocate all remaining to 8/9/10 among available free indices
    atk_real = [i for i in free_idx if (i + 1) in (8, 9, 10)]
    atk_pos = [free_idx.index(i) for i in atk_real]
    parts = rand_partition(remaining, len(atk_pos), min_each=0)
    random.shuffle(parts)
    for p, amt in zip(atk_pos, parts):
        x[p] += amt

    return fill_free(prefix, free_idx, x)

def gen_type2() -> List[int]:
    """
    (2) bone + main attack n piles (drop 10, keep 9),
        attacked sum > 27; others sprinkle small or 0.
    """
    prefix, free_idx, free_sum = make_bone()
    x = [0] * len(free_idx)

    # choose attacked from piles 1..9 excluding 10, must include 9
    possible = [i for i in free_idx if (i + 1) <= 9]
    pile9_idx = 8
    assert pile9_idx in free_idx
    n = random.randint(2, 5)
    attacked = set(random.sample(possible, min(n, len(possible))))
    attacked.add(pile9_idx)
    attacked = sorted(attacked)

    # sprinkle others, but leave at least 28 coins for attack_total
    for pos_i, real_i in enumerate(free_idx):
        if real_i not in attacked and random.random() < 0.4:
            x[pos_i] = sprinkle(3)

    # trim so sum(sprinkles) <= free_sum - 28
    x = trim_to_budget(x, free_sum - 28)
    remaining = free_sum - sum(x)

    # allocate attacked sum >= 28
    attack_total = random.randint(28, min(remaining, 95))
    rest = remaining - attack_total

    attacked_pos = [free_idx.index(i) for i in attacked]
    parts = rand_partition(attack_total, len(attacked_pos), min_each=0)
    random.shuffle(parts)
    for p, amt in zip(attacked_pos, parts):
        x[p] += amt

    # distribute rest anywhere (no cap)
    distribute_rest_with_cap(x, free_idx, rest)

    return fill_free(prefix, free_idx, x)

def gen_type3() -> List[int]:
    """(3) all 10 (does NOT obey bone by design)."""
    return [10] * P

def gen_type4() -> List[int]:
    """(4) bone + remaining numbers increasing-ish distribution."""
    prefix, free_idx, free_sum = make_bone()
    k = len(free_idx)

    # create increasing sequence for the free slots
    raw = sorted([random.random() for _ in range(k)])
    inc = [raw[0]] + [raw[i] - raw[i - 1] for i in range(1, k)]
    bids = []
    s = 0.0
    for d in inc:
        s += d + 0.08  # push increasing tendency
        bids.append(s)
    tot = sum(bids)
    alloc = [int(round(b / tot * free_sum)) for b in bids]

    # fix sum
    diff = free_sum - sum(alloc)
    while diff != 0:
        if diff > 0:
            j = random.randrange(k)
            alloc[j] += 1
            diff -= 1
        else:
            j = random.randrange(k)
            if alloc[j] > 0:
                alloc[j] -= 1
                diff += 1

    # ensure nondecreasing
    alloc = sorted(alloc)
    return fill_free(prefix, free_idx, alloc)

def gen_type5() -> List[int]:
    """
    (5) bone + main attack n piles (drop 9 and 10),
        attacked sum > 27; others sprinkle small or 0.
    """
    prefix, free_idx, free_sum = make_bone()
    x = [0] * len(free_idx)

    possible = [i for i in free_idx if (i + 1) <= 8]
    n = random.randint(2, 6)
    attacked = sorted(random.sample(possible, min(n, len(possible))))

    # sprinkle non-attacked (cap pile9/10 small at this stage too)
    for pos_i, real_i in enumerate(free_idx):
        pile = real_i + 1
        if real_i not in attacked and random.random() < 0.4:
            x[pos_i] = sprinkle(3)
        if pile in (9, 10):
            x[pos_i] = min(x[pos_i], random.randint(0, 2))

    # trim so we always have >=28 left for attack_total
    x = trim_to_budget(x, free_sum - 28)
    remaining = free_sum - sum(x)

    attack_total = random.randint(28, min(remaining, 95))
    rest = remaining - attack_total

    attacked_pos = [free_idx.index(i) for i in attacked]
    parts = rand_partition(attack_total, len(attacked_pos), min_each=0)
    random.shuffle(parts)
    for p, amt in zip(attacked_pos, parts):
        x[p] += amt

    # distribute rest, but keep piles 9/10 from getting too big
    distribute_rest_with_cap(
        x, free_idx, rest,
        cap_piles={9, 10},
        cap_value=3
    )

    return fill_free(prefix, free_idx, x)

def gen_type6() -> List[int]:
    """(6) bone + remaining numbers random (no extra structure)."""
    prefix, free_idx, free_sum = make_bone()
    alloc = rand_partition(free_sum, len(free_idx), min_each=0)
    random.shuffle(alloc)
    return fill_free(prefix, free_idx, alloc)

def make_opponents(seed: int = 1,
                   plan: List[Tuple[str, int]] = None) -> List[List[int]]:
    random.seed(seed)

    gens = {
        "t1": gen_type1,
        "t2": gen_type2,
        "t3": gen_type3,
        "t4": gen_type4,
        "t5": gen_type5,
        "t6": gen_type6,
    }

    if plan is None:
        plan = [("t1",10), ("t2",10), ("t3",10), ("t4",10), ("t5",10), ("t6",10)]

    opp = []
    for key, n in plan:
        for _ in range(n):
            opp.append(gens[key]())

    random.shuffle(opp)
    return opp

# -----------------------------
# Search "me" within bone space
# -----------------------------
def random_me_bone() -> List[int]:
    """Generate a random valid bone strategy (A or B)."""
    # same as type6 but for "me"
    prefix, free_idx, free_sum = make_bone()
    alloc = rand_partition(free_sum, len(free_idx), min_each=0)
    random.shuffle(alloc)
    return fill_free(prefix, free_idx, alloc)

def mutate_bone(x: List[int], steps: int = 50) -> List[int]:
    """
    Mutate while preserving bone validity + sum=100:
    move 1 coin among free indices only.
    """
    y = x[:]
    # detect which bone
    if y[0] == 11:
        free = list(range(1, P))
    else:
        free = list(range(2, P))

    for _ in range(steps):
        i = random.choice(free)
        j = random.choice(free)
        if i == j:
            continue
        if y[i] > 0:
            y[i] -= 1
            y[j] += 1

    # sanity
    assert sum(y) == TOTAL_COINS
    assert (y[0] == 11) or (y[0] == 1 and y[1] == 11)
    return y

def search_best(seed: int = 1,
                random_tries: int = 8000,
                climb_iters: int = 12000,
                print_opponents: bool = True) -> None:
    opponents = opponents = make_opponents(seed=2, plan=[
        ("t1", 100),
        ("t2", 100),
        ("t3", 100),
        ("t4", 100),
        ("t5", 100),
        ("t6", 200),
    ])
    print("=== Opponents generated ===")
    print("count =", len(opponents))
    if print_opponents:
        for i, o in enumerate(opponents, 1):
            print(f"opponent {i}: {o} sum={sum(o)}")

    # random search
    random.seed(seed + 999)
    best = None
    best_w = -1
    best_t = best_l = 0

    for _ in range(random_tries):
        cand = random_me_bone()
        w, t, l = wins_ties_losses(cand, opponents)
        if w > best_w:
            best, best_w, best_t, best_l = cand, w, t, l

    # hill climb
    cur = best[:]
    cur_w, cur_t, cur_l = best_w, best_t, best_l
    for _ in range(climb_iters):
        cand = mutate_bone(cur, steps=random.randint(10, 80))
        w, t, l = wins_ties_losses(cand, opponents)
        if w > cur_w:
            cur, cur_w, cur_t, cur_l = cand, w, t, l
            if w > best_w:
                best, best_w, best_t, best_l = cand, w, t, l

    print("\n=== Best found (within bone) ===")
    print("strategy:", best, "sum=", sum(best))
    bi = first_ge11_index(best)
    print("burn index (1..10):", (bi + 1) if bi >= 0 else -1)
    print(f"detail: wins/ties/losses = {best_w} {best_t} {best_l}  (out of {len(opponents)})")

    # Optional: show per-match result
    # Uncomment if you want full verification
    # print("\n=== Match details ===")
    # for i, o in enumerate(opponents, 1):
    #     sa, sb = match_score(best, o)
    #     res = "WIN" if sa > sb else ("LOSE" if sb > sa else "TIE")
    #     print(f"vs opponent {i}: score {sa}-{sb} -> {res}")

if __name__ == "__main__":
    # Tweak these if you want deeper search
    search_best(seed=1, random_tries=12000, climb_iters=20000, print_opponents=True)