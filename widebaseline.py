#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# import numpy
import cv2
import cv2.cv as cv
import time
import copy
import numpy as np
from numpy import arange,array,ones,linalg
import heapq
import math

"""
    This program to test combinations of feature detector, descriptor, and matchers
"""

def isGoodQuad(pts):
    hull = cv2.convexHull(pts, 0 ,1)#2nd Parameter: True= clockwise, 3rd Paramter: True to return points, False to return index 
    print hull
    if (len(hull)<4):
        print "concave quad"
        return False #when the quad is concave, there are less than 4 points to form the convex hull
    falseCount = 0
    for m in range(4):
        if (pts[m][0]!=hull[3-m][0][0] or pts[m][1]!=hull[3-m][0][1]):
            falseCount += 1
            break
    for m in range(4):
        if (pts[m][0]!=hull[m][0][0] or pts[m][1]!=hull[m][0][1]):
            falseCount += 1
            break
    if(falseCount == 2):
        print "false"
        return False
    return True

def match(detector_name, descriptor_name, matcher_name, image1_file, image2_file):
    
    print "\n###############################\n"
    print detector_name+"\t"+descriptor_name+"\t"+matcher_name
#     Read images
    image1                     = cv2.imread(image1_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    image2                     = cv2.imread(image2_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    #      -- Step 1: Compute keypoints      
    if (detector_name == "SIFT"):
        #configurable sift constructor -> http://docs.opencv.org/modules/nonfree/doc/feature_detection.html
        BEST_FEATURES       = 100  #default 0 meaning get all
        OCTAVE_LAYERS       = 5    #default 3
        CONTRAST_THRESHOLD  = 0.04 #default 0.04, larger -> less points
        EDGE_THRESHOLD      = 10   #default 10,   smaller -> less points
        SIGMA               = 1.6  #default 1.6.  meaning the level of gaussian blur
        detector            = cv2.SIFT(BEST_FEATURES,OCTAVE_LAYERS,CONTRAST_THRESHOLD,EDGE_THRESHOLD,SIGMA)
    elif (detector_name == "SURF"):
        HESSIAN_THRESHOLD   = 500  #larger -> less points
        OCTAVES             = 4    #default 4
        OCTAVE_LAYERS       = 2    #default 2
        EXTENDED            = True #default true, ie use 128 descriptor otherwise 64
        UPRIGHT             = False#default false, ie compute orientation
        detector            = cv2.SURF(HESSIAN_THRESHOLD,OCTAVES,OCTAVE_LAYERS,EXTENDED,UPRIGHT)        
    else:
        detector                  = cv2.FeatureDetector_create(detector_name)
    t1                         = time.time()
    
    keypoints1                 = detector.detect(image1)
    
    t2                         = time.time()
    print "Time to get keypoints for query image: ", str(t2 - t1)
    keypoints2                 = detector.detect(image2)

#   draw sift key points with sizes and their oritentations
    img1 = cv2.drawKeypoints(image1,keypoints1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints1.jpg',img1)
    img2 = cv2.drawKeypoints(image2,keypoints2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints2.jpg',img2)
    
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
    bimatches = matcher.knnMatch(descriptors1, descriptors2,2) #for each feature find 2 best matches if can.

    #Filter out a match if the two bests are too close. We'd rather not take those points.  
    RATIO_THREASHOLD = 0.9; # bigger -> more points remains, max = 1 ie don't discard any.
    matches = []
    for m in bimatches:
        if(len(m)==2):
            if(m[0].distance <= RATIO_THREASHOLD*m[1].distance):
                matches.append(m[0]) #knnmatch automatically sort the points such that first point always has less distance
        else:
            matches.append(m)

            
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
    h2, w2                   = image2.shape[:2]
    view1                    = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    view1[:h1, :w1]          = image1
    view1[:h2, w1:]          = image2
    view1[:, :]              = view1[:, :]
    view1[:, :]              = view1[:, :]
    view                     = copy.copy(view1)   
    view_trans_FM            = copy.copy(view)
    view_final_match_FM      = copy.copy(view)
    view_final_match_orig_FM = copy.copy(view)
    view_trans_HM            = copy.copy(view)
    view_final_match_HM      = copy.copy(view)
    view_final_match_orig_HM = copy.copy(view)
 

    
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
        Tpt   = (int(keypoints2[m.trainIdx].pt[0]+w1), int(keypoints2[m.trainIdx].pt[1]))

        src_points.append(keypoints1[m.queryIdx].pt)
        dest_points.append(keypoints2[m.trainIdx].pt)        

        cv2.line(view, Qpt, Tpt, color, 3)
        cv2.circle(view, Qpt, 10,color, 3)
        cv2.circle(view, Tpt, 10,color, 3)

    # Draw out good matches
    cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_good_matches.jpg", view)

    if len(src_points) > 8:
        #Compute the homography with RANSAC
        
        F, M  = cv2.findFundamentalMat(np.array(src_points, dtype='float32'), np.array(dest_points, dtype='float32'), cv.CV_FM_RANSAC, 3, 0.99)
        #stereoRectifyUncalibrated use different format for points.
        src_pts = []
        des_pts = []
        for i in range(len(src_points)):
            src_pts.append(src_points[i][0])
            src_pts.append(src_points[i][1])
            des_pts.append(dest_points[i][0])
            des_pts.append(dest_points[i][1])
    
        r, H1, H2 = cv2.stereoRectifyUncalibrated(np.array(src_pts, dtype='float32'), np.array(des_pts, dtype='float32'), F, image1.shape,threshold=5)
    
        srcTri        = np.array([(0,0), (w1,0), (w1,h1), (0,h1)],dtype='float32')
        srcTri        = np.array([srcTri])
        
        height, width = view.shape[:2] 
        desTri_FM     = cv2.perspectiveTransform(srcTri, H2) #Result from stereoRectifyUncalibrated
    
        if isGoodQuad(desTri_FM[0]):
        # if True:
    
            # //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            cv2.line(view_trans_FM, (int(desTri_FM[0][0][0]) + w1, int(desTri_FM[0][0][1])), (int(desTri_FM[0][1][0]) + w1, int(desTri_FM[0][1][1])), (255,255,255), 4)
            cv2.line(view_trans_FM, (int(desTri_FM[0][1][0]) + w1, int(desTri_FM[0][1][1])), (int(desTri_FM[0][2][0]) + w1, int(desTri_FM[0][2][1])), (255,255,255), 4)
            cv2.line(view_trans_FM, (int(desTri_FM[0][2][0]) + w1, int(desTri_FM[0][2][1])), (int(desTri_FM[0][3][0]) + w1, int(desTri_FM[0][3][1])), (255,255,255), 4)
            cv2.line(view_trans_FM, (int(desTri_FM[0][3][0]) + w1, int(desTri_FM[0][3][1])), (int(desTri_FM[0][0][0]) + w1, int(desTri_FM[0][0][1])), (255,255,255), 4)
        
            cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_perspectiveTrans_FM.jpg", view_trans_FM)
        
            # Perform perspectiveTransform on all source points;
            dest_trans_points_FM = cv2.perspectiveTransform(np.array([src_points], dtype='float32'), H2)
            final_src_points  = []
            final_des_points  = []
            #filter out miss matched points
            for i in range(len(src_points)):
                des_pt   = dest_points[i]
                trans_pt_FM = dest_trans_points_FM[0][i]
                if math.hypot(des_pt[0] - trans_pt_FM[0], des_pt[1] - trans_pt_FM[1]) < 200:
                    final_src_points.append(src_points[i])
                    final_des_points.append(((int(trans_pt_FM[0])), int(trans_pt_FM[1])))
        
                    color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
        
                    cv2.line(view_final_match_FM, (int(src_points[i][0]), int(src_points[i][1])), ((int(trans_pt_FM[0]))+w1, int(trans_pt_FM[1])), color, 3)
                    cv2.circle(view_final_match_FM, (int(src_points[i][0]), int(src_points[i][1])), 10,color, 3)
                    cv2.circle(view_final_match_FM, ((int(trans_pt_FM[0]))+w1, int(trans_pt_FM[1])), 10,color, 3)
        
                    #Draw original points on dest image
                    cv2.line(view_final_match_orig_FM, (int(src_points[i][0]), int(src_points[i][1])), ((int(des_pt[0]))+w1, int(des_pt[1])), color, 3)
        
                    cv2.circle(view_final_match_orig_FM, (int(src_points[i][0]), int(src_points[i][1])), 10,color, 3)
                    cv2.circle(view_final_match_orig_FM, ((int(des_pt[0]))+w1, int(des_pt[1])), 10,color, 3)
        
        
        
            print "Fundamental Metrix Final number of matches %d"% (len(final_src_points))
            if len(final_src_points) > 0:
                cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_FM_FinalMatches.jpg", view_final_match_FM)
                cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_FM_FinalMatches(orig).jpg", view_final_match_orig_FM)
    
    
        H,mask        = cv2.findHomography(np.array(src_points, dtype='float32'), np.array(dest_points, dtype='float32'), cv.CV_RANSAC)
        H             = np.array(H, dtype='float32')
        dest_trans_points_HM = cv2.perspectiveTransform(np.array([src_points], dtype='float32'), H)
        desTri_HM     = cv2.perspectiveTransform(srcTri, H) #result from homography
        if isGoodQuad(desTri_HM[0]):
        # if True:
    
            # //-- Draw lines between the corners (the mapped object in the scene - image_2 )    
            cv2.line(view_trans_HM, (int(desTri_HM[0][0][0]) + w1, int(desTri_HM[0][0][1])), (int(desTri_HM[0][1][0]) + w1, int(desTri_HM[0][1][1])), (0,204,0), 4)
            cv2.line(view_trans_HM, (int(desTri_HM[0][1][0]) + w1, int(desTri_HM[0][1][1])), (int(desTri_HM[0][2][0]) + w1, int(desTri_HM[0][2][1])), (0,204,0), 4)
            cv2.line(view_trans_HM, (int(desTri_HM[0][2][0]) + w1, int(desTri_HM[0][2][1])), (int(desTri_HM[0][3][0]) + w1, int(desTri_HM[0][3][1])), (0,204,0), 4)
            cv2.line(view_trans_HM, (int(desTri_HM[0][3][0]) + w1, int(desTri_HM[0][3][1])), (int(desTri_HM[0][0][0]) + w1, int(desTri_HM[0][0][1])), (0,204,0), 4)
        
            cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_perspectiveTrans_HM.jpg", view_trans_HM)
        
            # Perform perspectiveTransform on all source points;
            dest_trans_points_HM = cv2.perspectiveTransform(np.array([src_points], dtype='float32'), H)
            final_src_points  = []
            final_des_points  = []
            #filter out miss matched points
            for i in range(len(src_points)):
                des_pt   = dest_points[i]
                trans_pt_HM = dest_trans_points_HM[0][i]
                if math.hypot(des_pt[0] - trans_pt_HM[0], des_pt[1] - trans_pt_HM[1]) < 200:
                    final_src_points.append(src_points[i])
                    final_des_points.append(((int(trans_pt_HM[0])), int(trans_pt_HM[1])))
        
                    color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
        
                    cv2.line(view_final_match_HM, (int(src_points[i][0]), int(src_points[i][1])), ((int(trans_pt_HM[0]))+w1, int(trans_pt_HM[1])), color, 3)
                    cv2.circle(view_final_match_HM, (int(src_points[i][0]), int(src_points[i][1])), 10,color, 3)
                    cv2.circle(view_final_match_HM, ((int(trans_pt_HM[0]))+w1, int(trans_pt_HM[1])), 10,color, 3)
        
                    #Draw original points on dest image
                    cv2.line(view_final_match_orig_HM, (int(src_points[i][0]), int(src_points[i][1])), ((int(des_pt[0]))+w1, int(des_pt[1])), color, 3)
                    cv2.circle(view_final_match_orig_HM, (int(src_points[i][0]), int(src_points[i][1])), 10,color, 3)
                    cv2.circle(view_final_match_orig_HM, ((int(des_pt[0]))+w1, int(des_pt[1])), 10,color, 3)
        
        
            print "Homography found Final number of matches %d"% (len(final_src_points))
            if len(final_src_points) > 0:
                cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_HM_FinalMatches.jpg", view_final_match_HM)
                cv2.imwrite(detector_name+"_"+descriptor_name+"_"+matcher_name+"_HM_FinalMatches(orig).jpg", view_final_match_orig_HM)


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
