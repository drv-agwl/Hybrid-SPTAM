#!/usr/bin/env python

'''
stereo2depth - generate a depth map from a stereoscopic image using OpenCV.

Usage:
    Run `python stereo2depth.py images/stereo.jpg`
    and the depth map will be saved to `images/stereo_depth.jpg`
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import sys
import os as os
import json

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


# specific for ZED cam. Change
def rectifyFrames(left_image, right_image):
    lfx = 1399.56;
    lfy = 1399.56;
    lcx = 875.82;
    lcy = 565.139
    rfx = 1399.24;
    rfy = 1399.24;
    rcx = 898.46;
    rcy = 503.932

    # dist coeffs.
    ldist = np.array([-0.174868, 0.0277045, 0., 0., 0.])
    rdist = np.array([-0.176115, 0.0287941, 0., 0., 0.])
    # ldist = np.zeros(5); rdist = np.zeros(5)

    # Extrinsic parameters
    # rvec = np.eye(3)
    rvec = np.array([-0.0077, 0.0022, -0.0012])
    tvec = np.array([-0.120008, 0., 0.])

    # intrinsic matrix
    lK = np.array([[lfx, 0., lcx], [0., lfy, lcy], [0., 0., 1.]])
    rK = np.array([[rfx, 0., rcx], [0., rfy, rcy], [0., 0., 1.]])

    # image size
    size1 = left_image.shape[:2]
    size2 = right_image.shape[:2]
    im1_size = (size1[1], size1[0])
    im2_size = (size2[1], size2[0])

    data = cv2.stereoRectify(lK, ldist, rK, rdist, im1_size, rvec, tvec, alpha=0)
    R1, R2, P1, P2, Q, roi1, roi2 = data

    mapL = cv2.initUndistortRectifyMap(cameraMatrix=lK, distCoeffs=ldist,
                                       R=R1, newCameraMatrix=P1,
                                       size=im1_size, m1type=cv2.CV_8U)

    mapR = cv2.initUndistortRectifyMap(cameraMatrix=rK, distCoeffs=rdist,
                                       R=R2, newCameraMatrix=P2,
                                       size=im2_size, m1type=cv2.CV_8U)

    newLeft = cv2.remap(left_image, mapL[0], mapL[1], cv2.INTER_LINEAR)
    newRight = cv2.remap(right_image, mapR[0], mapR[1], cv2.INTER_LINEAR)

    return newLeft, newRight


def stereo2depth1(imgL, imgR):
    # Parameters from all steps are defined here to make it easier to adjust values.
    resolution = 1.0  # (0, 1.0]
    numDisparities = 16  # has to be dividable by 16
    blockSize = 5  # (0, 25]
    windowSize = 5  # Usually set equals to the block size
    filterCap = 63  # [0, 100]
    lmbda = 80000  # [80000, 100000]
    sigma = 1.2
    brightness = 0  # [-1.0, 1.0]
    contrast = 1  # [0.0, 3.0]

    # Step 1 - Load the input stereoscopic image
    # img = cv2.imread(filename)

    # Step 2 - Slice the input image into the left and right views.
    height, width = imgL.shape[:2]

    # Step 3 - Downsampling the images to the resolution level to speed up the matching at the cost of quality degradation.
    resL = cv2.resize(imgL, None, fx=resolution, fy=resolution, interpolation=cv2.INTER_AREA)
    resR = cv2.resize(imgR, None, fx=resolution, fy=resolution, interpolation=cv2.INTER_AREA)

    # Step 4 - Setup two stereo matchers to compute disparity maps both for left and right views.
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * 3 * windowSize ** 2,
        P2=32 * 3 * windowSize ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=filterCap,
        mode=cv2.STEREO_SGBM_MODE_HH
        # Run on HH mode which is more accurate than the default mode but much slower so it might not suitable for real-time scenario.
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Step 5 - Setup a disparity filter to deal with stereo-matching errors.
    #          It will detect inaccurate disparity values and invalidate them, therefore making the disparity map semi-sparse.
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Step 6 - Perform stereo matching to compute disparity maps for both left and right views.
    displ = left_matcher.compute(resL, resR)
    dispr = right_matcher.compute(resR, resL)

    # Step 7 - Perform post-filtering
    imgLb = cv2.copyMakeBorder(imgL, top=0, bottom=0, left=np.uint16(numDisparities / resolution), right=0,
                               borderType=cv2.BORDER_CONSTANT, value=[155, 155, 155])
    filteredImg = wls_filter.filter(displ, imgLb, None, dispr)

    # Step 8 - Adjust image resolution, brightness, contrast, and perform disparity truncation hack
    filteredImg = filteredImg * resolution
    filteredImg = filteredImg + (brightness / 100.0)
    filteredImg = (filteredImg - 128) * contrast + 128
    filteredImg = np.clip(filteredImg, 0, 255)
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.resize(filteredImg, (width, height), interpolation=cv2.INTER_CUBIC)  # Disparity truncation hack
    filteredImg = filteredImg[0:height, np.uint16(numDisparities / resolution):width]
    filteredImg = cv2.resize(filteredImg, (width, height), interpolation=cv2.INTER_CUBIC)  # Disparity truncation hack

    return filteredImg


def stereo2depth2(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. 
    Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 
    # 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image;
        # 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, \
                                beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


def extract_features(img, ftype='orb'):
    if ftype.lower() == 'orb':
        detector = cv2.ORB_create()
        extractor = detector
    elif ftype.lower() == 'brisk':
        detector = cv2.BRISK_create(20, 8)
        # detector = cv2.BRISK_create()
        extractor = detector

    keypoints = detector.detect(img)
    descriptors = extractor.compute(img, keypoints)
    return keypoints, descriptors[1]


def match_features(des1, des2):
    matches = []
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    bf = cv2.BFMatcher()
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pre_matches = bf.knnMatch(des1, des2, k=2)

    for m, n in pre_matches:
        if m.distance < 0.55 * n.distance:  # used: 0.65
            matches.append([m])
    return matches


def convert_matches2uv(matches, kp1, kp2):
    matched_uv1, matched_uv2 = [], []
    if len(matches) != 0:
        for i in range(0, len(matches)):
            p1 = kp1[matches[i][0].queryIdx].pt
            p2 = kp2[matches[i][0].trainIdx].pt
            matched_uv1.append(tuple(p1))
            matched_uv2.append(tuple(p2))
    return matched_uv1, matched_uv2


def compute_depth(depthMap, uvL, uvR, k):
    factor = 5
    depthMap = depthMap / factor
    h, w = depthMap.shape[:2]

    m2f = 3.28084

    depth1 = [];
    depth2 = []
    for i in range(len(uvL)):
        d1, d2 = None, None
        disp1 = uvL[i][0] - uvR[i][0]
        if disp1 > 0:
            d1 = k[0] * k[4] / disp1
            depth1.append(d1)
        # disp2 = depthMap[int(uvL[i][0])][int(uvL[i][1])]
        disp2 = depthMap[int(uvL[i][1])][int(uvL[i][0])]
        if disp2 > 0:
            d2 = k[0] * k[4] / w * disp2
            depth2.append(d2)
        # print(d1, d2)
        print('point:', i)
        d1a = d1 if d1 != None else 0
        d2a = d2 if d2 != None else 0
        # check if the point is within the bounding box of the detected semantic object
        print('d1: {}m {}ft'.format(d1, d1a * m2f))
        print('d2: {}m {}ft'.format(d2, d2a * m2f));
        print()

    print('--------------------------------------------------------------')
    print('Min depth (d1): {}m {}ft'.format(min(depth1), min(depth1) * m2f))
    print('Max depth (d1): {}m {}ft'.format(max(depth1), max(depth1) * m2f))
    return depth1, depth2


def find_kps_depths(img):
    height, width = img.shape[:2]
    imgL = img[0:height, 0:int(width / 2)]
    imgR = img[0:height, int(width / 2):]
    imgL, imgR = rectifyFrames(imgL, imgR)

    # k = [349.89, 349.89, 313.705, 193.03475, 0.120008] #ZED VGA
    # k = [718.856, 718.856, 607.1928, 185.2157, 0.5371657]
    k = [1399.56, 1399.56, 875.825, 565.139, 0.120008]

    depthMap_type = '1'

    if depthMap_type == '1':
        depthMap = stereo2depth1(imgL, imgR)
    elif depthMap_type == '2':
        depthMap = stereo2depth2(imgL, imgR)
    else:
        print('Wrong type! exiting')
        sys.exit()

    ftype = 'brisk'
    kp1, des1 = extract_features(imgL, ftype)
    kp2, des2 = extract_features(imgR, ftype)
    matches = match_features(des1, des2)
    uvL, uvR = convert_matches2uv(matches, kp1, kp2)

    # compute depth
    depth1, depth2 = compute_depth(depthMap, uvL, uvR, k)

    vals = []
    for i in range(len(depth1)):
        vals.append([uvL[i][0], uvL[i][1], depth1[i] * 1000])

    return vals


# if __name__ == '__main__':

#     path = './Oct_2020/ground_truth_images/rgb'
#     rectify = True
#     depth_dict = {}

#     for img_name in os.listdir(path):
#         print(img_name)
#         img = cv2.imread(os.path.join(path, img_name))

#         # Step 2 - Slice the input image into the left and right views.
#         height, width = img.shape[:2]
#         imgL = img[0:height, 0:int(width / 2)]
#         imgR = img[0:height, int(width / 2):]
#         if rectify:
#             imgL, imgR = rectifyFrames(imgL, imgR)

#         # k = [349.89, 349.89, 313.705, 193.03475, 0.120008] #ZED VGA
#         # k = [718.856, 718.856, 607.1928, 185.2157, 0.5371657]
#         k = [1399.56, 1399.56, 875.825, 565.139, 0.120008]

#         depthMap_type = '1'

#         if depthMap_type == '1':
#             depthMap = stereo2depth1(imgL, imgR)
#         elif depthMap_type == '2':
#             depthMap = stereo2depth2(imgL, imgR)
#         else:
#             print('Wrong type! exiting')
#             sys.exit()

#         ftype = 'brisk'
#         kp1, des1 = extract_features(imgL, ftype)
#         kp2, des2 = extract_features(imgR, ftype)
#         matches = match_features(des1, des2)
#         uvL, uvR = convert_matches2uv(matches, kp1, kp2)

#         # compute depth
#         depth1, depth2 = compute_depth(depthMap, uvL, uvR, k)

#         vals = []
#         for i in range(len(depth1)):
#             vals.append([uvL[i][0], uvL[i][1], depth1[i]*1000])

#         depth_dict[img_name] = vals
#         '''
#         #Generate point cloud
#         Q = np.float32([[1, 0, 0, -k[2]],
#                         [0, 1, 0, -k[3]],  # turn points 180 deg around x-axis,
#                         [0, 0, 0,  k[0]],  # so that y-axis looks up
#                         [0, 0, -1/k[4], 0]])
    
#         points = cv2.reprojectImageTo3D(depthMap, Q)
    
#         x ,y, z= [], [], []
#         for i in range(len(points)):
#             for j in range(len(points[0])):
#                 if points[i][j][0] == np.inf:
#                     continue
#                 x.append(points[i][j][0])
#                 y.append(points[i][j][1])
#                 z.append(points[i][j][2])
    
#         #select random points
#         idx = np.random.choice(range(len(x)), 2000, replace=False)
#         rx = [x[i] for i in range(len(x)) if i in idx]
#         ry = [y[i] for i in range(len(y)) if i in idx]
#         rz = [z[i] for i in range(len(z)) if i in idx]
    
#         fig = plt.figure()
#         ax = plt.axes(projection='3d')
#         ax.scatter3D(rx, rz, ry, c=rz, cmap='Greens')
#         ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
#         plt.show()'''

#         # show_img = True
#         # if show_img:
#         #     frame = imgL.copy();
#         #     c = (64, 34, 233)
#         #     for p in uvL:
#         #         coord = (int(p[0]), int(p[1]))
#         #         frame = cv2.circle(frame, coord, radius=5, color=c, thickness=2)
#         #
#         #     h, w = frame.shape[:2]
#         #     if h > 480:
#         #         frame = cv2.resize(frame, (int(w / 3), int(h / 3)))
#         #         depthMap = cv2.resize(depthMap, (int(w / 3), int(h / 3)))
#         #     cv2.imshow('rgb', frame)
#         #     cv2.imshow('depth', depthMap);  # cv2.waitKey(1000)
#         #
#         #     if cv2.waitKey() & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
#         #         cv2.destroyAllWindows()

#     with open('./depths.json', 'w') as f:
#         json.dump(depth_dict, f, indent=2)
