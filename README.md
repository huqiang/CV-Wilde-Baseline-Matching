## CS4243 Project Sem1 AY1314  
### wide baseline matching
by:  

Name 		| Matric Number
--- 		| ---
Chua Cai Jun| A0087796A
Hu Qiang 	| A0077857J
Li Yin 		| A0085686L
Liu Yang 	| A0077978B  

#### Development Environment
OpenCV Version: 2.4.6.1(Mac OS X)   2.4.6.0(Windows)  
Python Version: 2.7.*  

#### Running command on Linux Enviroment
```bash
./widebaseline.py QUERY_IMAGE TRAINED_IMAGE
python widebaseline.py QUERY_IMAGE TRAINED_IMAGE
```  

#### NOTE:  
+ Ths program is not tested on Python3
+ SIFT feature detector cannot be used in Windows Platform
+ Dont use same image as both QUERY_IMAGE and TRAINED_IMAGE  

#### General Idea:  
1. Detectect keypoints
1. Extract descriptors
1. Find matches
1. Keep good matches
1. Perform geo-verification using RANSAC
1. Further filter out no-so-good matchs based on found foundamental matrix or homography