# Documentation of Projection package

TODO

## Overview

TODO

## Example code

```python
import numpy as np
import src.projection as proj

calibrator = proj.RansacCalibrator(intrinsics=proj.Intrinsics(cx=0,
                                                              cy=0,
                                                              fx=1.5,
                                                              fy=1.5,
                                                              res=np.array([800, 600], dtype=float)),
                                   person_height=1750)
head_pixels = [..]
leg_pixels = [..]
camera = calibrator.calibrate(p_bottom=leg_pixels, p_top=head_pixels)
leg_positions = proj.back_project(pixels=leg_pixels, camera=camera)

```

## Dependencies

TODO

| File name             | Directory         | Version used                              | Download link                                                                                                                     |
|-----------------------|-------------------|-------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------|
|||||
