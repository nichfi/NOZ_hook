import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import cv2.aruco as aruco
# import meshio
# from meshpy.tet import MeshInfo, build

SCALING = 1
ARUCO_DICT = cv.aruco.DICT_5X5_50  #We are using 5 x 5 marker with id 9, so it is enough to try to match to the 50 first ids.
aruco_scale = 46.0

# CMTX and DIST obtained from calibrator function
CMTX = np.array(
    [7415.916112463534, 0.0, 1210.8355378298197, 0.0, 7399.646184729046, 1624.6141260308648, 0.0, 0.0, 1.0],
    dtype='float32').reshape(3, 3)
DIST = np.array(
    [-0.40646631101635533, 24.397052120119906, 0.0016761726863413812, -0.0002671261331697189, -294.68776899111657],
    dtype='float32')

# NOTE: if you get error that cv2.aruco does not exist:
# pip uninstall opencv-python
# pip install opencv-contrib-python
aruco_edges1 = np.array([[0, 0, 0], [aruco_scale, 0, 0], [aruco_scale, aruco_scale, 0], [0, aruco_scale, 0]],
                        dtype='float32').reshape((4, 1, 3))

aruco_edges2 = np.array([[136.5 + aruco_scale, 0, 0], [136.5 + 2 * aruco_scale, 0, 0],
                         [136.5 + 2 * aruco_scale, aruco_scale, 0], [136.5 + aruco_scale, aruco_scale, 0]],
                        dtype='float32').reshape((4, 1, 3))

aruco_edges3 = np.array(
    [[0, 293.2, 0], [aruco_scale, 293.2, 0], [aruco_scale, 293.2 + aruco_scale, 0], [0, 293.2 + aruco_scale, 0]],
    dtype='float32').reshape((4, 1, 3))

aruco_edges4 = np.array(
    [[136.5 + aruco_scale, 293.2, 0], [136.5 + 2 * aruco_scale, 293.2, 0],
     [136.5 + 2 * aruco_scale, 293.2 + aruco_scale, 0], [136.5 + aruco_scale, 293.2 + aruco_scale, 0]],
    dtype='float32').reshape((4, 1, 3))

aruco_edge_array = [aruco_edges1, aruco_edges2, aruco_edges3, aruco_edges4]


def get_aruco_coords(points):
    aruco_edges = np.array([[0, 0, 0], [0, aruco_scale, 0], [aruco_scale, 0, 0], [aruco_scale, 0, 0]],
                           dtype='float32').reshape((4, 1, 3))

    ret, rvec, tvec = cv.solvePnP(aruco_edges, points, CMTX, DIST)

    if ret:
        unitv_points = np.array([[0, 0, 0], [aruco_scale, 0, 0], [0, aruco_scale, 0], [0, 0, aruco_scale]],
                                dtype='float32').reshape((4, 1, 3))

        points, jac = cv.projectPoints(unitv_points, rvec, tvec, CMTX, DIST)
        #the returned points are pixel coordinates of each unit vector.
        return points, rvec, tvec

    #return empty arrays if rotation and translation values not found
    else:
        return [], [], []


def contours_world(fn):
    distance = 500
    print("Getting contours")
    contour_points = contours(fn)[::20]

    img = cv.imread(fn)
    img = resize(img, scaling=SCALING)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT)

    arucoParams = cv.aruco.DetectorParameters_create()
    corners, ids, rejected = cv.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    img = cv.cvtColor(gray_img, cv.COLOR_GRAY2RGB)  #Translate back to color image
    aruco_edges = np.array([[0, 0, 0], [0, aruco_scale, 0], [aruco_scale, aruco_scale, 0], [aruco_scale, 0, 0]],
                           dtype='float32').reshape((4, 1, 3))

    # The cv::solvePnP() returns the rotation and the translation vectors that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame
    ret, rvec, tvec = cv.solvePnP(aruco_edges, corners[0], CMTX, DIST)
    rmat = cv.Rodrigues(rvec)[0]


def resize(img, scaling=0.4):
    width = int(img.shape[1] * scaling)
    height = int(img.shape[0] * scaling)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def contours(fn):
    img = cv.imread(fn, 0)
    img = resize(img, scaling=SCALING)

    img = cv.GaussianBlur(img, (5, 5), 5)
    # img = cv.medianBlur(img,15)
    # img = cv.medianBlur(img,25)

    thresholded = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)
    ret, thresholded = cv.threshold(img, 130, 255, cv.THRESH_BINARY)

    # Inversion needed for contour detection to work (detect contours from zero background)
    thresholded = cv.bitwise_not(thresholded)

    contours, hierarchy = cv.findContours(image=thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    # get largest contour only
    contours = [max(contours, key=cv.contourArea)]

    coords = list(zip((list(contours[0][:, 0][:, 0])), (list(contours[0][:, 0][:, 1]))))
    # print(list(coords)[::10])
    return coords

    contour_img = img.copy()
    cv.drawContours(image=contour_img,
                    contours=contours,
                    contourIdx=-1,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv.LINE_AA)

    blank_image = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    blank_image.fill(255)

    only_contours = cv.drawContours(image=blank_image,
                                    contours=contours,
                                    contourIdx=-1,
                                    color=(0, 255, 0),
                                    thickness=2,
                                    lineType=cv.LINE_AA)

    images = [img, thresholded, contour_img, only_contours]
    titles = ['Original Image', 'Thresholded inverse', 'Contours on image', 'Contours']

    fig = plt.figure()

    for i in range(len(images)):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()
    # plt.draw()
    # plt.waitforbuttonpress(0)
    # plt.close(fig)


def main():

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    fns = glob.glob('hook_data/OnePlus/Photos_tele_aruco/*')
    ax.scatter([0], [0], [0], color="black", marker="x")
    camera_locs = []
    for fn in fns:
        c, coordinate_system_dirs, contour_points_world = contours_world(fn)
        x = [i[0] for i in contour_points_world]
        y = [i[1] for i in contour_points_world]
        z = [i[2] for i in contour_points_world]

        # ax.scatter(*c)
        for i in range(3):
            # print(*c, *coordinate_system_dirs[:, i])
            colors = ["red", "green", "blue"]
            ax.quiver(*c, *coordinate_system_dirs[:, i], color=colors[i])
            ax.scatter(x, y, z)
    ax.set_aspect('equal', adjustable='box')

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return
    camera_locs = np.array(camera_locs)


def test_multi_aruco(index):
    fns = glob.glob('hook_data/aruco_test3/1*')
    fns += glob.glob('hook_data/aruco_test2/1*')
    fns += glob.glob('hook_data/aruco_test1/1*')
    fns += glob.glob('hook_data/aruco_test1/2*')
    fns.sort()

    fn = fns[index]
    aruco_edges = aruco_edge_array[index]
    contour_points = contours(fn)[::200]

    img = cv.imread(fn)
    img = resize(img, scaling=SCALING)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
    arucoParams = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

    corners, ids, rejected_img_points = detector.detectMarkers(img)
    img = cv.cvtColor(gray_img, cv.COLOR_GRAY2RGB)
    ret, rvec, tvec = cv.solvePnP(aruco_edges, corners[0], CMTX, DIST)

    rmat = cv.Rodrigues(rvec)[0]
    cameraPosition = np.array(-np.matrix(rmat).T * np.matrix(tvec))
    # print("Camera position:")
    # print(cameraPosition)
    # print("rmat")
    # print(rmat)
    # print(tvec)
    # print(np.column_stack((rmat[:,0], rmat[:,1], tvec[:,0])))

    H = CMTX @ np.column_stack((rmat[:, 0], rmat[:, 1], tvec[:, 0]))
    image_points_in_world = [(H @ np.array([i[0], i[1], 1]) / 1) for i in contour_points]

    return cameraPosition, image_points_in_world


if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = ["red", "green", "blue", "orange"]
    for i in range(4):
        cameraPosition, image_points_in_world = test_multi_aruco(i)
        ax.scatter(*cameraPosition[:, 0])
        for ip in image_points_in_world:
            ax.scatter(*ip, marker=".", color=colors[i])

    for i in aruco_edge_array:
        for point in (i[:, 0]):
            ax.scatter(*point, color="black", marker="x")

    ax.set_aspect('equal', 'box')
    ax.elev = 60
    ax.azim = 0
    plt.show()
