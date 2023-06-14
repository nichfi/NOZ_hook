import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob

# import meshio
# from meshpy.tet import MeshInfo, build
RESIZE_FACTOR = 0.4


def get_qr_coords(points):
    qr_edges = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype='float32').reshape((4, 1, 3))

    # determine the orientation of QR code coordinate system with respect to camera coorindate system.

    # Calibration matrix for tele, scale 40%
    cmtx = np.array(
        [2.74776550e+03, 0.00000000e+00, 6.61091669e+02, 0.00000000e+00, 2.76467463e+03, 6.42984203e+02, 0.00000000e+00,
         0.00000000e+00, 1.00000000e+00], dtype='float32').reshape(3, 3)
    dist = np.array([-8.81347370e-02, 4.97392211e+00, 8.31318415e-03, 6.17847741e-03, -3.60252884e+01], dtype='float32')

    # Calibration matrix for main camera, scale 20%
    # cmtx = np.array([1.25555113e+03, 0.00000000e+00, 8.14196406e+02, 0.00000000e+00, 1.25976263e+03, 5.61741891e+02, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00], dtype='float32').reshape(3, 3)
    # dist = np.array([0.02182182, -0.39061909, -0.00225075,  0.00677493,  0.03312771], dtype='float32')

    ret, rvec, tvec = cv.solvePnP(qr_edges, points, cmtx, dist)

    if ret:
        unitv_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32').reshape((4, 1, 3))

        points, jac = cv.projectPoints(unitv_points, rvec, tvec, cmtx, dist)
        # the returned points are pixel coordinates of each unit vector.
        return points, rvec, tvec

    # return empty arrays if rotation and translation values not found
    else:
        return [], [], []


def resize(img, scaling=0.5):
    width = int(img.shape[1] * scaling)
    height = int(img.shape[0] * scaling)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def qr_orientation(fn):
    qr = cv.QRCodeDetector()
    img = cv.imread(fn)
    img = resize(img, RESIZE_FACTOR)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.medianBlur(img,3)
    # _, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    # img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 11, 2)

    points = np.array([-1] * 8).reshape(1, 4, 2)
    last_measurements = [5 * [points]]
    for i in range(40, 150, 2):
        _, img = cv.threshold(gray_img, i, 255, cv.THRESH_BINARY)
        ret_qr, points = qr.detect(img)
        print(i)
        if ret_qr:
            print("points", points)
            last_measurements.append(points)
            last_measurements.pop(0)
            avg_matrix = np.mean(np.array([i for i in last_measurements]), axis=0)
            if np.allclose(points, avg_matrix, rtol=5e-04, atol=5e-02):  # Require two similar detections
                break

    print(points)

    if ret_qr:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        axis_points, rvec, tvec = get_qr_coords(points)

        # BGR color format
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]

        # check axes points are projected to camera view.
        if len(axis_points) > 0:
            axis_points = axis_points.reshape((4, 2))

            origin = (int(axis_points[0][0]), int(axis_points[0][1]))

            for p, c in zip(axis_points[1:], colors[:3]):
                p = (int(p[0]), int(p[1]))
                cv.line(img, origin, p, c, 3)

    cv.imshow('frame', img)
    cv.waitKey(0)


def contours(fn):
    img = cv.imread(fn, 0)
    img = cv.medianBlur(img, 5)
    img = cv.medianBlur(img, 15)
    img = cv.medianBlur(img, 25)

    thresholded = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)
    ret, thresholded = cv.threshold(img, 130, 255, cv.THRESH_BINARY)

    # Inversion needed for contour detection to work (detect contours from zero background)
    thresholded = cv.bitwise_not(thresholded)

    contours, hierarchy = cv.findContours(image=thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    # get largest contour only
    contours = [max(contours, key=cv.contourArea)]

    print(contours)
    contour_img = img.copy()
    cv.drawContours(image=contour_img,
                    contours=contours,
                    contourIdx=-1,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv.LINE_AA
                    )

    blank_image = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    blank_image.fill(255)

    only_contours = cv.drawContours(image=blank_image,
                                    contours=contours,
                                    contourIdx=-1,
                                    color=(0, 255, 0),
                                    thickness=2,
                                    lineType=cv.LINE_AA
                                    )

    images = [img, thresholded, contour_img, only_contours]
    titles = ['Original Image', 'Adaptive mean thresholded', 'Contours on image', 'Contours']

    fig = plt.figure()

    for i in range(len(images)):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()
    # plt.draw()
    # plt.waitforbuttonpress(0)
    # plt.close(fig)


if __name__ == "__main__":
    # gen_mesh()
    fns = glob.glob('hook_data/hook_camera/QR*')

    fns = glob.glob('hook_data/OnePlus/down_scaled/SCREEN7.jpg')
    fns = glob.glob('hook_data/OnePlus/down_scaled/SCREEN*')
    fns = glob.glob('hook_data/OnePlus/Photos_tele/IMG20221109163538*')
    fns = glob.glob('hook_data/OnePlus/Photos_tele/*')
    # fns = glob.glob('hook_data/hook_camera/QR*')
    # fns = glob.glob('hook_data/OnePlus/IPAD8*')

    for fn in fns:
        print(fn)
        contours(fn)
        # qr_orientation(fn)
