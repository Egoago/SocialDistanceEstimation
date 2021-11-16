# Documentation of Tracking package

To use this package, `import tracking`

`tracking` contains `Tracker`, `Person`, `BBoxFilter`, `create_tracker`.

## Overview

`Tracker` is the interface of trackers. It has only one implementation at the moment, which is a wrapper
around a third party component called `motpy`.

`Person` represents a tracked person with a smoothed bbox and a color.

Call `create_tracker` to get a tracker. It needs the expected delta time between frames
and a `BBoxFilter` object for filtering.

`BBoxFilter` has many arguments with which it is customizable what bboxes do we want to keep and what to drop.

## Example code

```python
import tracking as tr

bbox_filter = tr.BBoxFilter(img_size=(640, 480),
                            min_aspect=0,
                            max_aspect=1,
                            max_rel_height=1
                            ...)
tracker = tr.create_tracker(dt = 30, bbox_filter=bbox_filter)
bboxes = [...]
people = tracker.track(bboxes)
for person in people:
    print(person.bbox)
    print(person.color)
```

## Dependencies

| Package name      | Purpose                                       |
|-------------------|-----------------------------------------------|
|   numpy           |    Used for data storage and manipulation.    |
|   motpy           |    Third party tracker using Kalman filters.  |

Dependencies can be installed in an anaconda environment with the next commands:
```commandline
conda install numpy
pip install motpy
```
It is important to leave all the pip dependencies to the end of the initialization process.