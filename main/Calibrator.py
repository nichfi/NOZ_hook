import os
import numpy as np
import cv2 as cv
import glob

CHESSBOARD_WIDTH = 10  # 9
CHESSBOARD_HEIGHT = 7  # 6
RESIZE_IMAGES = True
SCALING = 0.2

chess_size = 23.7

# IMAGES = glob.glob('hook_data/hook_camera/CHESS*')
# IMAGES = glob.glob('hook_data/OnePlus/down_scaled/CHESS*')
# IMAGES = glob.glob('hook_data/OnePlus/CHESS*') #Full-size. WARNING: takes approx. 15 min-MANY HOURS per image
# IMAGES = glob.glob('hook_data/test_images/down_scaled/size_4000/CHESS*') #Local testing. NOTE: This folder is git-ignored.
# IMAGES = glob.glob('hook_data/OnePlus/CHESS*')
IMAGES = glob.glob('hook_data/OnePlus/Calibration_tele/*')

print(IMAGES)


def resize_image(img):
    # copied from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    width = int(img.shape[1] * SCALING)
    height = int(img.shape[0] * SCALING)
    dim = (width, height)

    # resize image
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def calibrate_camera():
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHESSBOARD_WIDTH * CHESSBOARD_HEIGHT, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT] * 23.7).T.reshape(-1, 2)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fn in IMAGES:
        print(fn)
        img = cv.imread(fn)
        if RESIZE_IMAGES:
            img = resize_image(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), None)
        # If found, add object points, image points (after refining them)
        print(ret)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners2, ret)

            # cv.imshow('img', img)
            # cv.waitKey()

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Estimate error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(ret)
    print("total error: {}".format(mean_error / len(objpoints)))
    print("------")

    print("CMTX = np.array({}, dtype='float32').reshape(3, 3)".format(list(mtx.flatten())))
    print("DIST = np.array({}, dtype='float32')".format(list(dist[0])))

if __name__ == '__main__':
    calibrate_camera()
