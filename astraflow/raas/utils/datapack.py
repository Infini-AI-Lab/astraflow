import itertools
from typing import Any


def flat2d(arr: list[list[Any]]) -> list[Any]:
    return list(itertools.chain(*arr))
