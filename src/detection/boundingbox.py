from typing import NamedTuple, Tuple, List


def opencv2opengl(coords: Tuple[int, int], img_height: int) -> Tuple[int, int]:
    return coords[0], img_height-coords[1]


def opengl2opencv(coords: Tuple[int, int], img_height: int) -> Tuple[int, int]:
    return opencv2opengl(coords, img_height)


class BoundingBox(NamedTuple):
    """
    BoundingBox represents a 2D bounding box of an object. It holds the top-left coordinates (x and y) of the box and it's
        dimensions, w and h (along the axes x and y, respectively).
    BoundingBox acts as a tuple, but it's attributes can also be accessed by their names.

    Usage:
        Examples: `bb[0:1]`, `*bb`, but `bb.x` is also possible
    """
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def bottom(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h

    def top(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y

    def corners(self) -> List[int]:
        return [self.x, self.y, self.x + self.w, self.y + self.h]
