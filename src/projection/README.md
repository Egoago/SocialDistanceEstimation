# Documentation of Projection package

This package solves the calibration of the camera's extrinsic parameters
and the backprojection from pixel to the ground.

## Overview

To use this package, `import projection` as shown below.
Use-cases:
- **Calibration:** `Calibrator` interface has two implementations. `LinearCalibrator` contains the main algorithm, while
`RansacCalibrator` is a wrapper which calls *RANSAC* algorithm for outline detection to achieve a more robust solution.
It is slower, but can produce better results. To calibrate the camera, it needs the pixel coordinates of detected 
people's shoes and heads. The more data you can collect for calibration, the more precise the result will be
and it has to contain at least 3 different samples.
- **Back Projection:** You can project and back-project pixels and world coordinates using a calibrated camera.
See `project` and `back-project` functions.

## Example code

```python
import numpy as np
import projection as proj

calibrator = proj.RansacCalibrator(intrinsics=proj.Intrinsics(cx=0,
                                                              cy=0,
                                                              fx=1.5,
                                                              fy=1.5,
                                                              res=np.array([800, 600], dtype=float)),
                                   person_height=1750)
head_pixels = np.array([..])
leg_pixels = np.array([..])
camera = calibrator.calibrate(p_bottom=leg_pixels, p_top=head_pixels)
leg_positions = proj.back_project(pixels=leg_pixels, camera=camera)
```

## Dependencies

| Package name      | Purpose                                       |
|-------------------|-----------------------------------------------|
|   numpy           |    Used for data storage and manipulation.    |
|   scikit-skimage  |    From `skimage.measure` function `ransac`   |

Dependencies can be install in an anaconda environment with the next command:
```commandline
conda install numpy scikit-image
```
