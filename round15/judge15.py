import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

K = 10
START_COINS = 100
STRATEGY_LEN = 14

IDX_BUY_A = 0
IDX_BUY_B = 1
IDX_BUY_C = 2
IDX_INSURANCE = 3
IDX_PILES = 4

SKILL_COST_A = 2
SKILL_COST_B = 12
SKILL_COST_C = 15

EPS = 1e-9
round_no = 15
name = 'results_round15'


def skill_cost_from_flags(buy_a: int, buy_b: int, buy_c: int) -> int:
    return int(buy_a) * SKILL_COST_A + int(buy_b) * SKILL_COST_B + int(buy_c) * SKILL_COST_C


def strategy_skill_cost(x) -> int:
    return skill_cost_from_flags(int(x[IDX_BUY_A]), int(x[IDX_BUY_B]), int(x[IDX_BUY_C]))


def strategy_total_used(x) -> int:
    arr = np.asarray(x, dtype=np.int64)
    return strategy_skill_cost(arr) + int(arr[IDX_INSURANCE]) + int(arr[IDX_PILES:IDX_PILES + K].sum())


def format_strategy(x) -> str:
    arr = np.asarray(x, dtype=np.int64)
    return (
        f"A={int(arr[0])} B={int(arr[1])} C={int(arr[2])} insurance={int(arr[3])} "
        f"piles={arr[4:14].tolist()}"
    )


def _is_valid_strategy_py(x) -> bool:
    arr = np.asarray(x)
    if arr.shape != (STRATEGY_LEN,):
        return False

    buy_a = int(arr[IDX_BUY_A])
    buy_b = int(arr[IDX_BUY_B])
    buy_c = int(arr[IDX_BUY_C])
    insurance = int(arr[IDX_INSURANCE])
    piles = np.asarray(arr[IDX_PILES:IDX_PILES + K], dtype=np.int64)

    if buy_a not in (0, 1) or buy_b not in (0, 1) or buy_c not in (0, 1):
        return False
    if insurance < 0:
        return False
    if np.any(piles < 0):
        return False
    if buy_a == 0 and insurance != 0:
        return False
    if buy_a == 1 and insurance <= 0:
        return False
    if skill_cost_from_flags(buy_a, buy_b, buy_c) + insurance + int(piles.sum()) != START_COINS:
        return False
    return True


if NUMBA_OK:
    @njit(cache=True)
    def skill_cost_from_flags_nb(buy_a, buy_b, buy_c):
        return buy_a * SKILL_COST_A + buy_b * SKILL_COST_B + buy_c * SKILL_COST_C

    @njit(cache=True)
    def is_valid_strategy_nb(x):
        if x.shape[0] != STRATEGY_LEN:
            return False

        buy_a = int(x[IDX_BUY_A])
        buy_b = int(x[IDX_BUY_B])
        buy_c = int(x[IDX_BUY_C])
        insurance = int(x[IDX_INSURANCE])

        if (buy_a != 0 and buy_a != 1) or (buy_b != 0 and buy_b != 1) or (buy_c != 0 and buy_c != 1):
            return False
        if insurance < 0:
            return False

        total = skill_cost_from_flags_nb(buy_a, buy_b, buy_c) + insurance
        for i in range(K):
            v = int(x[IDX_PILES + i])
            if v < 0:
                return False
            total += v

        if buy_a == 0 and insurance != 0:
            return False
        if buy_a == 1 and insurance <= 0:
            return False
        if total != START_COINS:
            return False
        return True

    @njit(cache=True)
    def calculate_match_result_nb(a, b):
        a_valid = is_valid_strategy_nb(a)
        b_valid = is_valid_strategy_nb(b)
        if not a_valid or not b_valid:
            if a_valid and not b_valid:
                return 1
            if b_valid and not a_valid:
                return -1
            return 0

        buy_a_a = int(a[IDX_BUY_A])
        buy_b_a = int(a[IDX_BUY_B])
        buy_c_a = int(a[IDX_BUY_C])
        ins_a = int(a[IDX_INSURANCE])

        buy_a_b = int(b[IDX_BUY_A])
        buy_b_b = int(b[IDX_BUY_B])
        buy_c_b = int(b[IDX_BUY_C])
        ins_b = int(b[IDX_INSURANCE])

        final_a = np.empty(K, dtype=np.int64)
        final_b = np.empty(K, dtype=np.int64)
        c_flip_a = np.zeros(K, dtype=np.bool_)
        c_flip_b = np.zeros(K, dtype=np.bool_)

        bonus_a = 0.0
        bonus_b = 0.0
        best_loss_diff_a = 10 ** 9
        best_loss_idx_a = -1
        best_loss_diff_b = 10 ** 9
        best_loss_idx_b = -1

        for i in range(K):
            ai = int(a[IDX_PILES + i])
            bi = int(b[IDX_PILES + i])
            final_a[i] = ai
            final_b[i] = bi

            if ai < bi:
                diff = bi - ai
                if diff < best_loss_diff_a or (diff == best_loss_diff_a and i > best_loss_idx_a):
                    best_loss_diff_a = diff
                    best_loss_idx_a = i
                if buy_b_a == 1:
                    bonus_a += 0.15 * bi
                if buy_c_a == 1 and (diff == 1 or diff == 2):
                    c_flip_a[i] = True
            elif ai > bi:
                diff = ai - bi
                if diff < best_loss_diff_b or (diff == best_loss_diff_b and i > best_loss_idx_b):
                    best_loss_diff_b = diff
                    best_loss_idx_b = i
                if buy_b_b == 1:
                    bonus_b += 0.15 * ai
                if buy_c_b == 1 and (diff == 1 or diff == 2):
                    c_flip_b[i] = True

        if buy_a_a == 1 and ins_a > 0 and best_loss_idx_a >= 0:
            final_a[best_loss_idx_a] += ins_a
        if buy_a_b == 1 and ins_b > 0 and best_loss_idx_b >= 0:
            final_b[best_loss_idx_b] += ins_b

        score_a = bonus_a
        score_b = bonus_b
        for i in range(K):
            value = float(i + 1)
            if c_flip_a[i]:
                score_a += value
            elif c_flip_b[i]:
                score_b += value
            else:
                if final_a[i] > final_b[i]:
                    score_a += value
                elif final_a[i] < final_b[i]:
                    score_b += value

        if score_a > score_b + EPS:
            return 1
        if score_b > score_a + EPS:
            return -1
        return 0
else:
    skill_cost_from_flags_nb = None
    is_valid_strategy_nb = None
    calculate_match_result_nb = None


def calculate_match_result_round15(a, b) -> int:
    aa = np.asarray(a, dtype=np.int16)
    bb = np.asarray(b, dtype=np.int16)
    if NUMBA_OK:
        return int(calculate_match_result_nb(aa, bb))

    a_valid = _is_valid_strategy_py(aa)
    b_valid = _is_valid_strategy_py(bb)
    if not a_valid or not b_valid:
        if a_valid and not b_valid:
            return 1
        if b_valid and not a_valid:
            return -1
        return 0

    buy_a_a = int(aa[IDX_BUY_A])
    buy_b_a = int(aa[IDX_BUY_B])
    buy_c_a = int(aa[IDX_BUY_C])
    ins_a = int(aa[IDX_INSURANCE])
    piles_a = np.asarray(aa[IDX_PILES:IDX_PILES + K], dtype=np.int64)

    buy_a_b = int(bb[IDX_BUY_A])
    buy_b_b = int(bb[IDX_BUY_B])
    buy_c_b = int(bb[IDX_BUY_C])
    ins_b = int(bb[IDX_INSURANCE])
    piles_b = np.asarray(bb[IDX_PILES:IDX_PILES + K], dtype=np.int64)

    final_a = piles_a.copy()
    final_b = piles_b.copy()
    bonus_a = 0.0
    bonus_b = 0.0
    c_flip_a = np.zeros(K, dtype=np.bool_)
    c_flip_b = np.zeros(K, dtype=np.bool_)
    best_loss_diff_a = 10 ** 9
    best_loss_idx_a = -1
    best_loss_diff_b = 10 ** 9
    best_loss_idx_b = -1

    for i in range(K):
        ai = int(piles_a[i])
        bi = int(piles_b[i])
        if ai < bi:
            diff = bi - ai
            if diff < best_loss_diff_a or (diff == best_loss_diff_a and i > best_loss_idx_a):
                best_loss_diff_a = diff
                best_loss_idx_a = i
            if buy_b_a == 1:
                bonus_a += 0.15 * bi
            if buy_c_a == 1 and (diff == 1 or diff == 2):
                c_flip_a[i] = True
        elif ai > bi:
            diff = ai - bi
            if diff < best_loss_diff_b or (diff == best_loss_diff_b and i > best_loss_idx_b):
                best_loss_diff_b = diff
                best_loss_idx_b = i
            if buy_b_b == 1:
                bonus_b += 0.15 * ai
            if buy_c_b == 1 and (diff == 1 or diff == 2):
                c_flip_b[i] = True

    if buy_a_a == 1 and ins_a > 0 and best_loss_idx_a >= 0:
        final_a[best_loss_idx_a] += ins_a
    if buy_a_b == 1 and ins_b > 0 and best_loss_idx_b >= 0:
        final_b[best_loss_idx_b] += ins_b

    score_a = bonus_a
    score_b = bonus_b
    for i in range(K):
        value = i + 1
        if c_flip_a[i]:
            score_a += value
        elif c_flip_b[i]:
            score_b += value
        else:
            if final_a[i] > final_b[i]:
                score_a += value
            elif final_a[i] < final_b[i]:
                score_b += value

    if score_a > score_b + EPS:
        return 1
    if score_b > score_a + EPS:
        return -1
    return 0


calculate_match_result = calculate_match_result_round15
is_valid_strategy = _is_valid_strategy_py if not NUMBA_OK else lambda x: bool(
    is_valid_strategy_nb(np.asarray(x, dtype=np.int16)))


strategies15 = [
    np.array([0, 0, 0, 0, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10], dtype=np.int16),
    np.array([1, 0, 0, 8, 4, 4, 4, 4, 6, 8, 10, 12, 16, 22], dtype=np.int16),
    np.array([0, 1, 0, 0, 0, 0, 2, 4, 6, 8, 10, 16, 20, 24], dtype=np.int16),
    np.array([0, 0, 1, 0, 1, 1, 2, 3, 5, 7, 11, 15, 19, 24], dtype=np.int16),
    np.array([1, 0, 1, 6, 1, 1, 2, 3, 5, 7, 9, 12, 20, 20], dtype=np.int16),
    np.array([1, 1, 0, 5, 0, 0, 1, 2, 4, 8, 10, 16, 20, 22], dtype=np.int16),
    np.array([0, 1, 1, 0, 0, 1, 1, 2, 4, 6, 10, 14, 18, 22], dtype=np.int16),
]
# min_values15 = np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
min_values15 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
must_beat15 = np.array([
    [0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    # [1, 1, 0, 4, 0, 0, 0, 0, 6, 8, 13, 17, 17, 21],
    # [1, 0, 0, 5, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    # [1, 0, 0, 3, 0, 0, 4, 6, 8, 8, 13, 15, 18, 21],
    # [1, 1, 0, 15, 4, 5, 6, 8, 12, 15, 21, 0, 0, 0],
], dtype=np.int16)

strategies = [np.asarray(x, dtype=np.int16).copy() for x in strategies15]
min_values = min_values15.copy()
must_beat = must_beat15.copy()


# ===== Humanized random generator for round 15 =====

def _feasible_combos(min_piles: np.ndarray, min_insurance_when_a: int):
    combos = []
    for buy_a in (0, 1):
        for buy_b in (0, 1):
            for buy_c in (0, 1):
                skill_cost = skill_cost_from_flags(buy_a, buy_b, buy_c)
                min_ins = min_insurance_when_a if buy_a == 1 else 0
                min_total = skill_cost + min_ins + int(min_piles.sum())
                if min_total <= START_COINS:
                    combos.append((buy_a, buy_b, buy_c, skill_cost, min_ins))
    if not combos:
        raise ValueError(
            'No feasible skill combination under current costs/min_values')
    return combos


def _combo_weight(buy_a: int, buy_b: int, buy_c: int) -> float:
    # 让随机池更接近“会被人提交”的组合，而不是完全均匀
    table = {
        (0, 0, 0): 1.2,
        (1, 0, 0): 1.0,
        (0, 1, 0): 1.3,
        (0, 0, 1): 0.5,
        (1, 1, 0): 1.5,
        (1, 0, 1): 1.0,
        (0, 1, 1): 1.7,
        (1, 1, 1): 1.2,
    }
    return table[(buy_a, buy_b, buy_c)]


def _choose_combo(rng: np.random.Generator, combos):
    w = np.array([_combo_weight(c[0], c[1], c[2])
                 for c in combos], dtype=np.float64)
    w /= w.sum()
    idx = int(rng.choice(len(combos), p=w))
    return combos[idx]


def _sample_insurance_humanized(rng: np.random.Generator, budget_after_skills: int, min_pile_sum: int, min_ins: int) -> int:
    max_ins = budget_after_skills - min_pile_sum
    if max_ins < min_ins:
        raise ValueError('Infeasible insurance budget for skill A')

    # 大保险金通常不划算，强偏向 1~12，少量到 18，再少量放宽到上限
    hard_cap = min(max_ins, 18)
    soft_cap = min(max_ins, 12)

    u = float(rng.random())
    if u < 0.72:
        hi = max(min_ins, soft_cap)
        return int(rng.integers(min_ins, hi + 1))
    elif u < 0.94:
        lo = max(min_ins, min(6, hard_cap))
        return int(rng.integers(lo, hard_cap + 1))
    else:
        # 极少数才允许更大，保留探索性
        return int(rng.integers(min_ins, max_ins + 1))


def _sample_focus_count(rng: np.random.Generator) -> int:
    # 更像人：常见主攻 3~5 个堆，少数 2 或 6 个
    vals = np.array([2, 3, 4, 5, 6], dtype=np.int64)
    probs = np.array([0.08, 0.28, 0.34, 0.22, 0.08], dtype=np.float64)
    return int(rng.choice(vals, p=probs))


def _sample_focus_indices(rng: np.random.Generator, count: int) -> np.ndarray:
    # 偏好高价值堆
    weights = np.array([1, 1, 1, 2, 3, 5, 7, 9, 11, 13], dtype=np.float64)
    weights /= weights.sum()
    idx = rng.choice(np.arange(K), size=count, replace=False, p=weights)
    idx.sort()
    return idx.astype(np.int64)


def _template_weights(template_id: int) -> np.ndarray:
    # 模板化，而不是均匀 Dirichlet(1)
    if template_id == 0:
        # 后段重压
        return np.array([0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.8, 4.0, 5.2, 6.3], dtype=np.float64)
    if template_id == 1:
        # 7-10 双峰/三峰重压
        return np.array([0.2, 0.3, 0.5, 0.6, 0.8, 1.2, 3.5, 4.8, 5.0, 5.5], dtype=np.float64)
    if template_id == 2:
        # 阶梯上升
        return np.array([0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.9, 3.7, 4.6], dtype=np.float64)
    if template_id == 3:
        # 中后段均衡，但高段仍更重
        return np.array([0.4, 0.4, 0.6, 0.8, 1.2, 1.7, 2.1, 2.5, 3.0, 3.3], dtype=np.float64)
    # 稀疏锚点型：基础较小，后续再加 focus
    return np.array([0.3, 0.3, 0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 2.8], dtype=np.float64)


def _sample_piles_humanized(rng: np.random.Generator, remain: int, min_piles: np.ndarray, buy_a: int, buy_b: int, buy_c: int) -> np.ndarray:
    piles = min_piles.astype(np.int16).copy()
    if remain <= 0:
        return piles

    # 技能不同，倾向不同
    template_ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    if buy_b == 1 and buy_c == 1:
        probs = np.array([0.20, 0.20, 0.14, 0.18, 0.28], dtype=np.float64)
    elif buy_b == 1:
        probs = np.array([0.16, 0.20, 0.18, 0.16, 0.30], dtype=np.float64)
    elif buy_c == 1:
        probs = np.array([0.16, 0.18, 0.26, 0.20, 0.20], dtype=np.float64)
    else:
        probs = np.array([0.28, 0.24, 0.18, 0.18, 0.12], dtype=np.float64)
    template_id = int(rng.choice(template_ids, p=probs))

    base_w = _template_weights(template_id)

    # 稀疏重点堆：让结果少一点“均匀胡撒”
    focus_cnt = _sample_focus_count(rng)
    focus_idx = _sample_focus_indices(rng, focus_cnt)
    w = base_w.copy()
    for idx in focus_idx:
        # 在重点堆上额外放大
        w[idx] *= float(rng.uniform(1.6, 3.8))

    # 低价值堆轻微抑制，避免前面太平均
    low_cap = int(min(8, remain * 0.18))
    high_floor = int(min(12, remain * 0.10))

    p = rng.dirichlet(w)
    extra = rng.multinomial(remain, p).astype(np.int16)

    # 二次修饰：把一部分低堆金币挪给后段/重点堆，更像人
    if remain >= 20:
        for low in range(0, 4):
            if extra[low] > 0 and int(extra[:4].sum()) > low_cap:
                take = min(int(extra[low]), int(
                    rng.integers(0, 1 + extra[low] // 2)))
                if take > 0:
                    extra[low] -= np.int16(take)
                    target = int(rng.choice(focus_idx if len(
                        focus_idx) > 0 else np.arange(6, 10)))
                    extra[target] += np.int16(take)

    # 保证后段通常有一定厚度
    if int(extra[7:].sum()) < high_floor and remain >= high_floor:
        need = high_floor - int(extra[7:].sum())
        for low in range(0, 7):
            if need <= 0:
                break
            mv = min(int(extra[low]), need)
            if mv > 0:
                extra[low] -= np.int16(mv)
                target = int(rng.integers(7, 10))
                extra[target] += np.int16(mv)
                need -= mv

    piles += extra
    return piles.astype(np.int16)


def sample_random_strategy(rng: np.random.Generator, min_values=None) -> np.ndarray:
    mins = np.zeros(STRATEGY_LEN, dtype=np.int16) if min_values is None else np.asarray(
        min_values, dtype=np.int16).copy()
    if mins.shape != (STRATEGY_LEN,):
        raise ValueError(
            f'min_values must have shape ({STRATEGY_LEN},), got {mins.shape}')

    min_piles = np.maximum(mins[IDX_PILES:IDX_PILES + K], 0).astype(np.int16)
    min_insurance_when_a = max(1, int(mins[IDX_INSURANCE]))

    combos = _feasible_combos(min_piles, min_insurance_when_a)
    buy_a, buy_b, buy_c, skill_cost, min_ins = _choose_combo(rng, combos)

    x = np.zeros(STRATEGY_LEN, dtype=np.int16)
    x[IDX_BUY_A] = buy_a
    x[IDX_BUY_B] = buy_b
    x[IDX_BUY_C] = buy_c

    budget_after_skills = START_COINS - skill_cost
    if buy_a == 1:
        insurance = _sample_insurance_humanized(
            rng, budget_after_skills, int(min_piles.sum()), min_ins)
    else:
        insurance = 0
    x[IDX_INSURANCE] = insurance

    remain = START_COINS - skill_cost - insurance - int(min_piles.sum())
    x[IDX_PILES:IDX_PILES +
        K] = _sample_piles_humanized(rng, remain, min_piles, buy_a, buy_b, buy_c)

    if not _is_valid_strategy_py(x):
        # 回退：极少数情况下直接用简单合法方案，保证不会炸
        remain = START_COINS - skill_cost - insurance - int(min_piles.sum())
        p = rng.dirichlet(
            np.array([0.4, 0.4, 0.6, 0.9, 1.3, 1.8, 2.4, 3.2, 4.0, 4.8], dtype=np.float64))
        extra = rng.multinomial(remain, p).astype(np.int16)
        x[IDX_PILES:IDX_PILES + K] = min_piles + extra

    return x.astype(np.int16)
