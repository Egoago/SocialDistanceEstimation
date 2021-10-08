from collections import namedtuple

"""
BoundingBox acts as a tuple.
Usage:
    Examples: `bb[0:1]` or `*bb`
"""
BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'h', 'w'])
