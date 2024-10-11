from functools import reduce
from math import gcd
from time import time
import numpy as np
from typing import List, Tuple
from functools import wraps


def split_num(num: int, parts: int = None) -> List[int]:
    """
        Split number num to parts partitions
        parts: if None, parts will be a random value
    """
    if parts is None:
        parts = np.random.randint(1, num + 1)
    elif num < parts:
        raise ValueError("Error: parts is larger than num")
    res = []
    for i in range(parts - 1):
        x = np.random.randint(1, num - parts + i + 2)
        num = num - x
        res.append(x)
    res.append(num)
    return res


def find_gcd(numbers) -> int:
    return reduce(gcd, numbers)


def find_factors(n):
    return set(reduce(list.__add__, ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def find_factors_tuple(num_list: List[Tuple[int, ...]]) -> List[List[Tuple[int, ...]]]:
    """
        [(4, 8), (2, 4)] -> [[(1, 2), (2, 4), (4, 8)], [(1, 2), (2, 4)]]
    """
    assert len(set(map(len, num_list))) == 1  # same tuple size
    factors_tuple = []
    for num in num_list:
        gcd = find_gcd(num)
        factors = find_factors(gcd)
        ft = []
        for f in factors:
            ft.append(tuple(n // f for n in num))
        factors_tuple.append(ft)
    return factors_tuple


def find_divisors_tuple(num_list: List[Tuple[int, int]]) -> List[List[int]]:
    """
        [(4, 8), (2, 4)] -> [[4, 2, 1], [2, 1]]
    """
    divisors_tuple = []
    for num in num_list:
        gcd = find_gcd(num)
        factors = find_factors(gcd)
        divisors_tuple.append(list(factors))
    return divisors_tuple


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'Function {f.__name__} took {te-ts:2.4f} seconds')
        return result

    return wrap


