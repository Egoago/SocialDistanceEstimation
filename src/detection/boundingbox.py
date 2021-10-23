from typing import NamedTuple

"""
BoundingBox represents a 2D bounding box of an object. It holds the top-left coordinates (x and y) of the box and it's
    dimensions, w and h (along the axes x and y, respectively).
BoundingBox acts as a tuple.
Usage:
    Examples: `bb[0:1]` or `*bb`
"""
BoundingBox = NamedTuple('BoundingBox', x=int, y=int, w=int, h=int)
