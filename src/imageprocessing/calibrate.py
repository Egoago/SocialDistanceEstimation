import cv2
import numpy as np
import pathlib
import glob
import os

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
def get_camera_params(dir_path, image_format, square_size, width, height, file='files/images_calibration/calibration_save.npy'):
    '''
        square_size: in mm
    '''

    data = None

    # if calibration file exist try to load
    if file is not None and os.path.isfile(file):
        try:
            data = load_calibration(file)
            print('Loaded calibration from file.')
        except(IOError):
            pass

    # if calibration file didn't exist, run calibration process
    if data is None:
        # data = ret, mtx, dist, rvecs, tvecs
        data = run_calibration(dir_path, image_format, square_size, width, height)

    ret, mtx, dist, rvecs, tvecs = data

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    return newcameramtx, mtx, dist


def load_calibration(file):
    data = np.load(file, allow_pickle=True)
    return data


def run_calibration(dir_path, image_format, square_size, width, height):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = pathlib.Path(dir_path).glob(f'*.{image_format}')
    # Iterate through all images
    for fname in images:
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        # print(ret, corners)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.save('files/images_calibration/calibration_save.npy', np.asarray([ret, mtx, dist, rvecs, tvecs], dtype=object))
    print('Calibration saved to file.')
    # data1 = np.load('calibration_save.npy',allow_pickle=True)
    return ret, mtx, dist, rvecs, tvecs
