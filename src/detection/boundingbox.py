from typing import NamedTuple

"""
BoundingBox acts as a tuple.
Usage:
    Examples: `bb[0:1]` or `*bb`
"""
BoundingBox = NamedTuple('BoundingBox', x=int, y=int, w=int, h=int)
