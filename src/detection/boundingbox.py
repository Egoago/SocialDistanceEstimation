from typing import NamedTuple

"""
BoundingBox represents a 2D bounding box of an object. It holds the top-left coordinates (x and y) of the box and it's
    dimensions, w and h (along the axes x and y, respectively).
BoundingBox acts as a tuple, but it's attributes can also be accessed by their names.

Usage:
    Examples: `bb[0:1]`, `*bb`, but `bb.x` is also possible
"""
BoundingBox = NamedTuple('BoundingBox', x=int, y=int, w=int, h=int)
