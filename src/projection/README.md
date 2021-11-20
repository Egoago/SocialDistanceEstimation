# Documentation of Projection package

This package solves the calibration of the camera's extrinsic parameters
and the backprojection from pixel to the ground.

## Overview

To use this package, `import projection` as shown below.
Use-cases:
- **Calibration:** `Calibrator` interface has two implementations. `LinearCalibrator` contains a linear algorithm,
with exact solution, while `LeastSquaresCalibrator` is a wrapper for a least squares problem which is solves with the 
Levenberg-Marquadt algorithm. Both implementations can be used with *Ransac* algorithm implemented in `RansacCalibrator`
for outline detection to achieve a more robust solution, but the `LeastSquaresCalibrator` performs worse with *Ransac*
and it is slower. To calibrate the camera, it needs the pixel coordinates of detected people's shoes and heads.
The more data you can collect for calibration, the more precise the result will be, and it has to contain at least
3 different samples.
- **Back Projection:** You can project and back-project pixels and world coordinates using a calibrated camera.
See `project` and `back-project` functions. **Important** to point out, that the projection and calibration modules use
opengl style image indexing, meaning that the (0,0) pixel is in tha left bottom corner. To convert between opencv and
opengl style indexes use `opencv2opengl` and `opengl2opencv` functions.

## Example code

```python
import numpy as np
import projection as proj

calibrator = proj.create_calibrator(intrinsics: proj.Intrinsics,
                      use_ransac=False,
                      person_height=1720.0,
                      method='least_squares'):
head_pixels = np.array([..])
leg_pixels = np.array([..])
camera = calibrator.calibrate(p_bottom=leg_pixels, p_top=head_pixels)
for pixel in leg_pixels:
    point = proj.back_project(np.array(proj.opencv2opengl(pixel, img_height)), camera)
    pixel_new = proj.opengl2opencv(tuple(proj.project(point, camera)[0]), img_height)
```

## Dependencies

| Package name      | Purpose                                           |
|-------------------|---------------------------------------------------|
|   numpy           |    Used for data storage and manipulation.        |
|   scikit-skimage  |    From `skimage.measure` function `ransac`       |
|   scipy           |    From `scipy.optimize` function `least_squares` |

Dependencies can be install in an anaconda environment with the next command:
```commandline
conda install numpy scikit-image scipy
```
