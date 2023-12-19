import cv2
from cv2 import resize
import numpy as np
from matplotlib import pyplot as plt
from traceback2 import print_tb
# import dlib

from utils import rotate_bbox, rotate_bound, warpImg, is_two_image_same
from utils import resizeImage, get_warpPerspective, get_angle_and_box_coord


def siftMatching(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print("Total good matches:", len(good))
    good = good[:20]
    return kp1, kp2, good


def main(imagePath):
    template = cv2.imread("test/CCCD8.jpg")
    sample = cv2.imread(f"train/{imagePath}.jpg")

    MIN_MATCH_COUNT = 20

    img1 = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)  # trainImage
    img2 = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)  # queryImage
    img1 = resizeImage(img1)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)

    img1 = cv2.filter2D(img1, -1, kernel=kernel)
    img2 = cv2.filter2D(img2, -1, kernel=kernel)
    img2 = cv2.normalize(img2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    kp1, kp2, good = siftMatching(img1, img2)

    if len(good) >= MIN_MATCH_COUNT:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        M2, mask2 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w, _ = img1.shape

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # border = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3)

        warped_image = get_warpPerspective(img2, M2, dst)

        (heigth_q, width_q) = img2.shape[:2]
        (cx, cy) = (width_q // 2, heigth_q // 2)

        angle, box = get_angle_and_box_coord(dst)

        rotated_img = rotate_bound(img2, angle)

        new_bbox = rotate_bbox(box, cx, cy, heigth_q, width_q, angle)

        warp_image = warpImg(rotated_img, new_bbox, heigth_q, width_q)
        # code gen ra chữ

        if (warp_image is not None):
            warp_image = cv2.cvtColor(warp_image, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(warp_image, 120, 255, cv2.THRESH_BINARY)
    return thresh1
    # plt.title("homography transformed image")
    # plt.imshow(warp_image)
    # plt.savefig('CCCD_test.png')
    # plt.show()