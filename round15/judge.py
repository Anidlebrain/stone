import numpy as np

try:
    from numba import njit
    NUMBA_OK = False
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
SKILL_COST_B = 10
SKILL_COST_C = 12

EPS = 1e-9


# =========================
# Round 15 helpers
# Strategy format:
# [buyA, buyB, buyC, insurance, p1, p2, ..., p10]
# =========================

def skill_cost_from_flags(buy_a: int, buy_b: int, buy_c: int) -> int:
    return int(buy_a) * SKILL_COST_A + int(buy_b) * SKILL_COST_B + int(buy_c) * SKILL_COST_C


def strategy_skill_cost(x) -> int:
    return skill_cost_from_flags(int(x[IDX_BUY_A]), int(x[IDX_BUY_B]), int(x[IDX_BUY_C]))


def strategy_total_used(x) -> int:
    return strategy_skill_cost(x) + int(x[IDX_INSURANCE]) + int(np.asarray(x[IDX_PILES:IDX_PILES + K]).sum())


def is_valid_strategy(x) -> bool:
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


def format_strategy(x) -> str:
    arr = np.asarray(x, dtype=np.int64)
    return (
        f"A={int(arr[0])} B={int(arr[1])} C={int(arr[2])} insurance={int(arr[3])} "
        f"piles={arr[4:14].tolist()}"
    )


def sample_random_strategy(rng: np.random.Generator, min_values=None) -> np.ndarray:
    mins = np.zeros(STRATEGY_LEN, dtype=np.int16) if min_values is None else np.asarray(min_values, dtype=np.int16).copy()
    if mins.shape != (STRATEGY_LEN,):
        raise ValueError(f'min_values must have shape ({STRATEGY_LEN},), got {mins.shape}')

    min_piles = np.maximum(mins[IDX_PILES:IDX_PILES + K], 0).astype(np.int16)
    min_insurance_when_a = max(1, int(mins[IDX_INSURANCE]))

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
        raise ValueError('No feasible skill combination under current costs/min_values')

    buy_a, buy_b, buy_c, skill_cost, min_ins = combos[int(rng.integers(0, len(combos)))]
    x = np.zeros(STRATEGY_LEN, dtype=np.int16)
    x[IDX_BUY_A] = buy_a
    x[IDX_BUY_B] = buy_b
    x[IDX_BUY_C] = buy_c

    budget_after_skills = START_COINS - skill_cost
    if buy_a == 1:
        max_ins = budget_after_skills - int(min_piles.sum())
        if max_ins < min_ins:
            raise ValueError('Infeasible insurance budget for skill A')
        insurance = int(rng.integers(min_ins, max_ins + 1))
    else:
        insurance = 0
    x[IDX_INSURANCE] = insurance

    remain = START_COINS - skill_cost - insurance - int(min_piles.sum())
    p = rng.dirichlet(np.ones(K))
    extra = rng.multinomial(remain, p).astype(np.int16)
    x[IDX_PILES:IDX_PILES + K] = min_piles + extra
    return x.astype(np.int16)


def calculate_match_result_round15(a, b) -> int:
    a = np.asarray(a)
    b = np.asarray(b)

    a_valid = is_valid_strategy(a)
    b_valid = is_valid_strategy(b)
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
    piles_a = np.asarray(a[IDX_PILES:IDX_PILES + K], dtype=np.int64)

    buy_a_b = int(b[IDX_BUY_A])
    buy_b_b = int(b[IDX_BUY_B])
    buy_c_b = int(b[IDX_BUY_C])
    ins_b = int(b[IDX_INSURANCE])
    piles_b = np.asarray(b[IDX_PILES:IDX_PILES + K], dtype=np.int64)

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


# =========================
# Round 15 config
# =========================

strategies15 = [
    np.array([0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
    np.array([1, 0, 0, 8, 4, 4, 4, 4, 6, 8, 10, 12, 16, 22], dtype=np.int16),
    np.array([0, 1, 0, 0, 0, 0, 2, 4, 6, 8, 10, 16, 20, 24], dtype=np.int16),
    np.array([0, 0, 1, 0, 1, 1, 2, 3, 5, 7, 11, 15, 19, 24], dtype=np.int16),
    np.array([1, 0, 1, 6, 1, 1, 2, 3, 5, 7, 9, 12, 20, 20], dtype=np.int16),
    np.array([1, 1, 0, 5, 0, 0, 1, 2, 4, 8, 10, 16, 20, 22], dtype=np.int16),
    np.array([0, 1, 1, 0, 0, 1, 1, 2, 4, 6, 10, 14, 18, 22], dtype=np.int16),
]

min_values15 = np.zeros(STRATEGY_LEN, dtype=np.int16)
must_beat15 = np.zeros((0, STRATEGY_LEN), dtype=np.int16)


# generic config helpers reused by t / t-2 / t-3 / t-4

def _zero_min_values():
    return np.zeros(STRATEGY_LEN, dtype=np.int16)


def _empty_must_beat():
    return np.zeros((0, STRATEGY_LEN), dtype=np.int16)


def _normalize_strategy_list(items):
    return [np.asarray(x, dtype=np.int16).copy() for x in items]


def _normalize_must_beat(items):
    if items is None:
        return _empty_must_beat()
    arr = np.asarray(items, dtype=np.int16)
    if arr.size == 0:
        return _empty_must_beat()
    if arr.ndim == 1:
        if arr.shape[0] != STRATEGY_LEN:
            raise ValueError(
                f"must_beat strategy length must be {STRATEGY_LEN}, got {arr.shape[0]}"
            )
        arr = arr.reshape(1, STRATEGY_LEN)
    if arr.ndim != 2 or arr.shape[1] != STRATEGY_LEN:
        raise ValueError(
            f"must_beat must have shape (n, {STRATEGY_LEN}), got {arr.shape}"
        )
    return arr.astype(np.int16)


ROUND_CONFIG = {
    15: {
        'match_func': calculate_match_result_round15,
        'strategies': strategies15,
        'min_values': min_values15,
        'must_beat': must_beat15,
    },
}

CURRENT_ROUND = 15


def get_round_config(round_id=None):
    rid = CURRENT_ROUND if round_id is None else int(round_id)
    if rid not in ROUND_CONFIG:
        raise ValueError(f'Unknown round: {rid}')
    raw = ROUND_CONFIG[rid]
    return {
        'round_no': rid,
        'name': f'results_round{rid}',
        'match_func': raw['match_func'],
        'strategies': _normalize_strategy_list(raw.get('strategies', [])),
        'min_values': np.asarray(raw.get('min_values', _zero_min_values()), dtype=np.int16).copy(),
        'must_beat': _normalize_must_beat(raw.get('must_beat', _empty_must_beat())),
    }


_config = get_round_config()
round_no = _config['round_no']
name = _config['name']
match_func = _config['match_func']
strategies = _config['strategies']
min_values = _config['min_values']
must_beat = _config['must_beat']

calculate_match_result = match_func
