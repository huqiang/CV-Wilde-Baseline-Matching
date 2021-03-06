#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# import numpy
import cv2, cv
import time
import copy
import numpy as np
from numpy import arange,array,ones,linalg
import heapq
import math

"""
    This program to test combinations of feature detector, descriptor, and matchers
"""

def match(detector_name, descriptor_name, matcher_name, image1_file, image2_file):
    
    print "\n###############################\n"
    print detector_name
#     Read images
    image1                     = cv2.imread(image1_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    image2                     = cv2.imread(image2_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    #      -- Step 1: Compute keypoints
    detector                   = cv2.FeatureDetector_create(detector_name)
    
    t1                         = time.time()
    
    keypoints1                 = detector.detect(image1)
    
    t2                         = time.time()
    print "Time to get keypoints for query image: ", str(t2 - t1)
    keypoints2                 = detector.detect(image2)
    
    #      -- Step 2: Compute descriptors
    descriptor                 = cv2.DescriptorExtractor_create(descriptor_name)
    
    t1                         = time.time()
    (keypoints1, descriptors1) = descriptor.compute(image1, keypoints1)
    t2                         = time.time()
    print "Time to get descriptors for query image: ", str(t2 - t1)
    (keypoints2, descriptors2) = descriptor.compute(image2, keypoints2)
    
#===============================================================================
# #     -- Step 3: Matching descriptor 
    t1 = time.time()
    matcher = cv2.DescriptorMatcher_create(matcher_name)
    matches = matcher.match(descriptors1, descriptors2)
#     
#     t2 = time.time()
#     
# #     print number of matches
#     print "time: ", t2-t1
#     print "detector: ", detector_name, " descriptor extractor: ", descriptor_name, " matcher: ", matcher_name
#     print '#matches:', len(matches)
#===============================================================================
    t3 = time.time()
    print "Time to match: ", str(t3  - t1)
#     -- Step 4: Draw matches on image

    print image1.shape
    h1, w1          = image1.shape[:2]
    print w1, h1
    h2, w2          = image2.shape[:2]
    view1           = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    view1[:h1, :w1] = image1
    view1[:h2, w1:] = image2
    view1[:, :]     = view1[:, :]
    view1[:, :]     = view1[:, :]
    
    
    view            = copy.copy(view1)
    
    view2           = copy.copy(view)
    view3           = copy.copy(view)
    view4           = copy.copy(view)

    
#     Draw all matches between two images
    for m in matches:
        color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
        Qpt   = (int(keypoints1[m.queryIdx].pt[0]), int(keypoints1[m.queryIdx].pt[1]))
        Tpt   = (int(keypoints2[m.trainIdx].pt[0])+w1, int(keypoints2[m.trainIdx].pt[1]))
        
        cv2.line(view1, Qpt, Tpt, color, 3)
        cv2.circle(view1, Qpt, 10,color, 3)
        cv2.circle(view1, Tpt, 10,color, 3)   
        
    cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_all_matches.jpg", view1)     

    dist     = [m.distance for m in matches]
    
    min_dist = min(dist)
    avg_dist = (sum(dist) / len(dist))
    print 'distance: min: %.3f' % min_dist
    print 'distance: mean: %.3f' % avg_dist
    print 'distance: max: %.3f' % max(dist)
    
    # threshold: half the mean
    # thres_dist = (sum(dist) / len(dist)) * 0.5
    
    # keep only the reasonable matches
    # good_matches = heapq.nsmallest(20, matches, key=lambda match: match.distance)
    good_matches = [m for m in matches if m.distance < avg_dist*0.8]
    # good_matches.sort(cmp=None, key=distance, reverse=False);
    # sorted(good_matches, key=lambda match: match.distance)
    
    print "Number of match is: "+str(len(matches))
    print "Number of good match is: "+str(len(good_matches))
    
    src_points  = []
    dest_points = []
    for m in good_matches:
        color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
        Qpt   = (int(keypoints1[m.queryIdx].pt[0]), int(keypoints1[m.queryIdx].pt[1]))
        Tpt   = (int(keypoints2[m.trainIdx].pt[0]), int(keypoints2[m.trainIdx].pt[1]))

        src_points.append(keypoints1[m.queryIdx].pt)
        dest_points.append(keypoints2[m.trainIdx].pt)        

        cv2.line(view, Qpt, (Tpt[0]+w1, Tpt[1]), color, 3)
        cv2.circle(view, Qpt, 10,color, 3)
        cv2.circle(view, (Tpt[0]+w1, Tpt[1]), 10,color, 3)

    # Draw out good matches
    cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_good_matches.jpg", view)

    #Compute the homography with RANSAC
    t1            = time.time()
    H,mask        = cv2.findHomography(np.array(src_points, dtype='float32'), np.array(dest_points, dtype='float32'), cv.CV_RANSAC)
    t3            = time.time()
    print "Time to perform geoverification: ", str(t3  - t1)
    
    F, M  = cv2.findFundamentalMat(np.array(src_points, dtype='float32'), np.array(dest_points, dtype='float32'), cv.CV_FM_RANSAC, 3, 0.99)

    src_pts = []
    des_pts = []
    for i in range(len(src_points)):
        src_pts.append(src_points[i][0])
        src_pts.append(src_points[i][1])
        des_pts.append(dest_points[i][0])
        des_pts.append(dest_points[i][1])

    r, H1, H2 = cv2.stereoRectifyUncalibrated(np.array(src_pts, dtype='float32'), np.array(des_pts, dtype='float32'), F, image1.shape,threshold=5)
    # r, H1, H2 = cv2.stereoRectifyUncalibrated(np.array((1,2,2,3,3,4,4,5), dtype='float32'), np.array((1,2,2,3,3,4,4,5), dtype='float32'), F, (2,2))

    H             = np.array(H, dtype='float32')
    srcTri        = np.array([(0,0), (w1,0), (w1,h1), (0,h1)],dtype='float32')
    srcTri        = np.array([srcTri])
    
    height, width = view.shape[:2] 
    desTri        = cv2.perspectiveTransform(srcTri, H2)

    # //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv2.line(view2, (int(desTri[0][0][0]) + w1, int(desTri[0][0][1])), (int(desTri[0][1][0]) + w1, int(desTri[0][1][1])), (255,255,255), 4)
    cv2.line(view2, (int(desTri[0][1][0]) + w1, int(desTri[0][1][1])), (int(desTri[0][2][0]) + w1, int(desTri[0][2][1])), (255,255,255), 4)
    cv2.line(view2, (int(desTri[0][2][0]) + w1, int(desTri[0][2][1])), (int(desTri[0][3][0]) + w1, int(desTri[0][3][1])), (255,255,255), 4)
    cv2.line(view2, (int(desTri[0][3][0]) + w1, int(desTri[0][3][1])), (int(desTri[0][0][0]) + w1, int(desTri[0][0][1])), (255,255,255), 4)

    cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_perspectiveTrans.jpg", view2)

    # Perform perspectiveTransform on all source points;
    dest_trans_points = cv2.perspectiveTransform(np.array([src_points], dtype='float32'), H2)
    final_src_points  = []
    final_des_points  = []
    #filter out miss matched points
    for i in range(len(src_points)):
        des_pt   = dest_points[i]
        trans_pt = dest_trans_points[0][i]
        if math.hypot(des_pt[0] - trans_pt[0], des_pt[1] - trans_pt[1]) < 100:
            final_src_points.append(src_points[i])
            final_des_points.append(((int(trans_pt[0])), int(trans_pt[1])))

            color = tuple([np.random.randint(0, 255) for _ in xrange(3)])

            cv2.line(view3, (int(src_points[i][0]), int(src_points[i][1])), ((int(trans_pt[0]))+w1, int(trans_pt[1])), color, 3)
            cv2.circle(view3, (int(src_points[i][0]), int(src_points[i][1])), 10,color, 3)
            cv2.circle(view3, ((int(trans_pt[0]))+w1, int(trans_pt[1])), 10,color, 3)

            #Draw original points on dest image
            cv2.line(view4, (int(src_points[i][0]), int(src_points[i][1])), ((int(des_pt[0]))+w1, int(des_pt[1])), color, 3)
            cv2.circle(view4, (int(src_points[i][0]), int(src_points[i][1])), 10,color, 3)
            cv2.circle(view4, ((int(des_pt[0]))+w1, int(des_pt[1])), 10,color, 3)
 
    print "Final number of matches %d"% (len(final_src_points))
    cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_FinalMatches.jpg", view3)
    cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_FinalMatches(orig).jpg", view4)

    print "\n###############################\n"
    
 
if __name__ == '__main__':
    match("ORB", "ORB", "BruteForce", sys.argv[1], sys.argv[2])
    # match("FAST", "ORB", "BruteForce", sys.argv[1], sys.argv[2])
    match("BRISK", "BRISK", "BruteForce", sys.argv[1], sys.argv[2])
    match("SURF", "SIFT", "FlannBased", sys.argv[1], sys.argv[2])
    match("SURF", "SIFT", "BruteForce", sys.argv[1], sys.argv[2])
    match("SIFT", "SIFT", "FlannBased", sys.argv[1], sys.argv[2])
    # match("SURF", "SIFT", "BruteForce-Hamming", sys.argv[1], sys.argv[2])
    #     match("BRISK", "BRISK", "FlannBased", sys.argv[1], sys.argv[2])
