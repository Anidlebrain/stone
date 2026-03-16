import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

K = 10
START_COINS = 100


def calculate_match_result_round8(a, b) -> int:
    win_count_a = 0
    win_count_b = 0
    value_a = 0
    value_b = 0

    for i in range(10):
        pile_value = i + 1
        if a[i] > b[i]:
            win_count_a += 1
            value_a += pile_value
        elif a[i] < b[i]:
            win_count_b += 1
            value_b += pile_value

    if win_count_a > win_count_b:
        return 1
    if win_count_b > win_count_a:
        return -1
    if value_a > value_b:
        return 1
    if value_b > value_a:
        return -1
    return 0


def calculate_match_result_round7(a, b) -> int:
    pairs = [(0, 9), (1, 8), (2, 7), (3, 6), (4, 5)]
    score_a = 0
    score_b = 0

    for i, j in pairs:
        a_win_i = a[i] > b[i]
        b_win_i = b[i] > a[i]
        a_win_j = a[j] > b[j]
        b_win_j = b[j] > a[j]

        a_wins_in_group = int(a_win_i) + int(a_win_j)
        b_wins_in_group = int(b_win_i) + int(b_win_j)
        max_value = j + 1

        if a_wins_in_group == 1:
            score_a += max_value
        elif a_wins_in_group == 2:
            score_a += 11

        if b_wins_in_group == 1:
            score_b += max_value
        elif b_wins_in_group == 2:
            score_b += 11

    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


strategies5 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
    np.array([9, 11, 13, 12, 15, 0, 0, 0, 20, 20], dtype=np.int16),
]


def calculate_match_result_round5(a, b) -> int:
    loss_a = 0
    loss_b = 0
    score_a = 0
    score_b = 0

    for i in range(10):
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

    if score_a > score_b:
        return 1
    if score_a < score_b:
        return -1
    return 0


def calculate_match_result_round6(a, b) -> int:
    streak_a = 0
    streak_b = 0
    score_a = 0
    score_b = 0

    for i in range(10):
        if a[i] > b[i]:
            score_a += i + 1
            streak_a += 1
            streak_b = 0
        elif a[i] < b[i]:
            score_b += i + 1
            streak_b += 1
            streak_a = 0
        else:
            streak_a = 0
            streak_b = 0

        if streak_a == 3:
            score_a += sum(range(i + 2, 11))
            break
        if streak_b == 3:
            score_b += sum(range(i + 2, 11))
            break

    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


strategies8 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
]

strategies7 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
]

strategies6 = [
    np.array([33, 33, 33, 1, 0, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([1, 0, 33, 0, 0, 33, 0, 0, 33, 0], dtype=np.int16),
    np.array([0, 0, 34, 33, 33, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
    np.array([1, 1, 91, 1, 1, 1, 1, 1, 1, 1], dtype=np.int16),
    np.array([25, 25, 50, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([0, 0, 50, 25, 25, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([20, 20, 60, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([0, 0, 60, 20, 20, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([30, 30, 40, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([0, 0, 40, 30, 30, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([1, 1, 51, 1, 1, 20, 1, 1, 20, 1], dtype=np.int16),
    np.array([1, 1, 42, 1, 1, 24, 1, 1, 17, 17], dtype=np.int16),
    np.array([20, 20, 40, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([1, 1, 40, 19, 19, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([12, 12, 76, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([0, 0, 76, 12, 12, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([0, 0, 76, 12, 12, 0, 0, 0, 0, 0], dtype=np.int16),
]


def calculate_match_result_round1(a, b) -> int:
    return calculate_match_result_round9(a, b)


def calculate_match_result_round9(a, b) -> int:
    value_a = 0
    value_b = 0
    for i in range(10):
        pile_value = i + 1
        if a[i] > b[i]:
            value_a += pile_value
        elif a[i] < b[i]:
            value_b += pile_value
    if value_a > value_b:
        return 1
    if value_b > value_a:
        return -1
    return 0


strategies9 = [
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 91], dtype=np.int16),
    np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 82], dtype=np.int16),
]


def calculate_match_result_round10(a, b) -> int:
    return calculate_match_result_round9(a, b)


strategies10 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
]

strategies11 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 100], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 45, 55, 0], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 34, 33, 33, 0], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 20, 20, 20, 20, 20], dtype=np.int16),
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 91], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 49, 51], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 30, 30, 40], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 15, 20, 30, 35], dtype=np.int16),
    np.array([5, 5, 5, 5, 5, 5, 10, 15, 20, 25], dtype=np.int16),
]


def calculate_match_result_round11(a, b) -> int:
    coins_a = START_COINS
    coins_b = START_COINS
    value_a = 0
    value_b = 0

    for i in range(10):
        ai = int(a[i])
        bi = int(b[i])

        if ai > bi:
            diff = ai - bi
            coins_a -= diff
            if coins_a < 0:
                return -1
            value_a += i + 1
        elif ai < bi:
            diff = bi - ai
            coins_b -= diff
            if coins_b < 0:
                return 1
            value_b += i + 1

    if value_a > value_b:
        return 1
    if value_b > value_a:
        return -1
    return 0


strategies12 = [
    np.array([2, 2, 2, 2, 2, 6, 12, 18, 24, 30], dtype=np.int16),
    np.array([2, 2, 2, 2, 2, 2, 10, 16, 24, 32], dtype=np.int16),
    np.array([2, 2, 2, 2, 2, 2, 2, 12, 24, 36], dtype=np.int16),
    np.array([2, 2, 2, 2, 2, 2, 8, 14, 20, 26], dtype=np.int16),
    np.array([2, 2, 2, 2, 2, 2, 2, 2, 18, 28], dtype=np.int16),
    np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 20], dtype=np.int16),
    np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 15, 25, 35], dtype=np.int16),
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
]


def calculate_match_result_round12(a, b) -> int:
    value_a = 0.0
    value_b = 0.0

    for i in range(10):
        pile_value = i + 1
        if a[i] > b[i]:
            value_a += pile_value
        elif a[i] < b[i]:
            value_b += pile_value

    used_a = 0
    used_b = 0
    for i in range(10):
        used_a += int(a[i])
        used_b += int(b[i])

    if used_a > START_COINS or used_b > START_COINS:
        if used_a > START_COINS and used_b > START_COINS:
            return 0
        return -1 if used_a > START_COINS else 1

    score_a = value_a + 0.5 * (START_COINS - used_a)
    score_b = value_b + 0.5 * (START_COINS - used_b)

    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


def calculate_match_result_round13(a, b) -> int:
    odd_bonus_map = {1: 0.1, 3: 0.2, 5: 0.3, 7: 0.4, 9: 0.5}
    base_a = 0.0
    base_b = 0.0
    bonus_a = 0.0
    bonus_b = 0.0

    for i in range(10):
        k = i + 1
        if a[i] > b[i]:
            if k % 2 == 0:
                base_a += k
            else:
                bonus_a += odd_bonus_map[k]
        elif b[i] > a[i]:
            if k % 2 == 0:
                base_b += k
            else:
                bonus_b += odd_bonus_map[k]

    score_a = base_a * (1.0 + bonus_a)
    score_b = base_b * (1.0 + bonus_b)

    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


strategies13 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
    np.array([4, 14, 4, 14, 6, 14, 8, 14, 10, 12], dtype=np.int16),
    np.array([3, 12, 3, 12, 5, 12, 7, 14, 18, 14], dtype=np.int16),
    np.array([2, 8, 2, 10, 6, 10, 16, 16, 18, 12], dtype=np.int16),
    np.array([2, 4, 2, 6, 4, 8, 10, 18, 20, 26], dtype=np.int16),
    np.array([1, 3, 2, 6, 6, 12, 14, 16, 18, 22], dtype=np.int16),
    np.array([8, 6, 10, 8, 12, 10, 14, 10, 16, 6], dtype=np.int16),
    np.array([0, 8, 2, 8, 6, 12, 12, 14, 18, 20], dtype=np.int16),
]

must_beat13 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16),
    np.array([2, 3, 2, 10, 4, 21, 5, 2, 12, 39], dtype=np.int16),
    np.array([4, 4, 7, 11, 11, 22, 16, 2, 21, 2], dtype=np.int16),
    np.array([0, 0, 0, 21, 1, 0, 0, 36, 0, 42], dtype=np.int16),
]

min_values13 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int16)

min_values14 = np.array([0, 0, 0, 0, 2, 4, 6, 8, 10, 10], dtype=np.int16)


def calculate_match_result_round14(a, b) -> int:
    score_a = 0
    score_b = 0
    wins_a = 0
    wins_b = 0

    for i in range(10):
        pile_value = i + 1
        if a[i] > b[i]:
            if wins_a < wins_b:
                score_a += pile_value * 2
            else:
                score_a += pile_value
            wins_a += 1
        elif a[i] < b[i]:
            if wins_b < wins_a:
                score_b += pile_value * 2
            else:
                score_b += pile_value
            wins_b += 1

    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


strategies14 = [
    np.array([0, 1, 2, 3, 6, 12, 16, 18, 20, 22], dtype=np.int16),
    np.array([1, 1, 2, 3, 6, 9, 14, 18, 22, 24], dtype=np.int16),
    np.array([2, 3, 4, 5, 8, 12, 14, 16, 18, 18], dtype=np.int16),
    np.array([5, 5, 5, 7, 8, 10, 13, 15, 16, 16], dtype=np.int16),
    np.array([4, 7, 4, 7, 5, 9, 13, 16, 17, 18], dtype=np.int16),
    np.array([6, 6, 6, 6, 8, 10, 12, 14, 16, 16], dtype=np.int16),
    np.array([10] * 10, dtype=np.int16),
    np.array([2, 2, 2, 2, 2, 2, 2, 2, 0, 92], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 50, 50], dtype=np.int16),
    np.array([12, 12, 12, 12, 12, 12, 12, 16, 0, 0], dtype=np.int16),
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 92, 0], dtype=np.int16),
]

min_values14 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16)
must_beat14 = [
    np.array([10] * 10, dtype=np.int16),
    np.array([2, 2, 2, 2, 2, 2, 2, 2, 0, 92], dtype=np.int16),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 50, 50], dtype=np.int16),
    np.array([12, 12, 12, 12, 12, 12, 12, 16, 0, 0], dtype=np.int16),
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 92, 0], dtype=np.int16),
    
]


def _zero_min_values():
    return np.zeros(K, dtype=np.int16)


def _empty_must_beat():
    return np.zeros((0, K), dtype=np.int16)


def _normalize_strategy_list(items):
    return [np.asarray(x, dtype=np.int16).copy() for x in items]


def _normalize_must_beat(items):
    if items is None:
        return _empty_must_beat()
    arr = np.asarray(items, dtype=np.int16)
    if arr.size == 0:
        return _empty_must_beat()
    if arr.ndim == 1:
        if arr.shape[0] != K:
            raise ValueError(
                f"must_beat strategy length must be {K}, got {arr.shape[0]}")
        arr = arr.reshape(1, K)
    if arr.ndim != 2 or arr.shape[1] != K:
        raise ValueError(
            f"must_beat must have shape (n, {K}), got {arr.shape}")
    return arr.astype(np.int16)


ROUND_CONFIG = {
    1: {"match_func": calculate_match_result_round1, "strategies": [], "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    5: {"match_func": calculate_match_result_round5, "strategies": strategies5, "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    6: {"match_func": calculate_match_result_round6, "strategies": strategies6, "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    7: {"match_func": calculate_match_result_round7, "strategies": strategies7, "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    8: {"match_func": calculate_match_result_round8, "strategies": strategies8, "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    9: {"match_func": calculate_match_result_round9, "strategies": strategies9, "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    10: {"match_func": calculate_match_result_round10, "strategies": strategies10, "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    11: {"match_func": calculate_match_result_round11, "strategies": strategies11, "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    12: {"match_func": calculate_match_result_round12, "strategies": strategies12, "min_values": _zero_min_values(), "must_beat": _empty_must_beat()},
    13: {"match_func": calculate_match_result_round13, "strategies": strategies13, "min_values": min_values13, "must_beat": must_beat13},
    14: {"match_func": calculate_match_result_round14, "strategies": strategies14, "min_values": min_values14, "must_beat": must_beat14},
}

CURRENT_ROUND = 14


def get_round_config(round_id=None):
    rid = CURRENT_ROUND if round_id is None else int(round_id)
    if rid not in ROUND_CONFIG:
        raise ValueError(f"Unknown round: {rid}")
    raw = ROUND_CONFIG[rid]
    return {
        "round_no": rid,
        "name": f"results_round{rid}",
        "match_func": raw["match_func"],
        "strategies": _normalize_strategy_list(raw.get("strategies", [])),
        "min_values": np.asarray(raw.get("min_values", _zero_min_values()), dtype=np.int16).copy(),
        "must_beat": _normalize_must_beat(raw.get("must_beat", _empty_must_beat())),
    }


_config = get_round_config()
round_no = _config["round_no"]
name = _config["name"]
match_func = _config["match_func"]
strategies = _config["strategies"]
min_values = _config["min_values"]
must_beat = _config["must_beat"]

if NUMBA_OK:
    calculate_match_result = njit(cache=True)(match_func)
else:
    calculate_match_result = match_func
