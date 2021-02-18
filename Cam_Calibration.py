import numpy as np
import cv2 
import glob
import os

# !!! PUT THIS SCRIPT INTO A FILE WHERE YOUR CALIBRATION IMAGES(.JPG otherwise correct line 23) ARE, THEN RUN !!!

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
# termination criteria
edge_of_square = 24 # 24 milimeter
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, edge_of_square, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Get the current working directory 
cwd = os.getcwd() 
#path_of_images = "C:/Users/..../Desktop/"
path_of_images = cwd
filename = path_of_images +"/*.jpg"

cap = cv2.VideoCapture(0)

print("{} image found.".format(len(os.listdir(path_of_images))))
images = glob.glob(filename)
for fname in images:
    distorted_img = cv2.imread(fname)
    gray = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(distorted_img, (9,6), corners2, ret)
        #cv2.imshow('distorted_img', distorted_img)
        #cv2.waitKey(500)
        
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save mtx and dist values into a txt file
with open(path_of_images+"/calibration_data.txt", "a") as file:
    np.savetxt(file, mtx, delimiter=',', header="mtx: ",)
    np.savetxt(path_of_images+"/calibration_data.txt", dist, delimiter=',', header="dist: ",)
    file.close()
print("camera matrix and distortion coefficients are saved")

# Calculate Re-projection error 
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
print ("Re-projection error: ", total_error/len(objpoints))

#**********************************************Sample of undistorted image********************************************
# undistort one of distorted images
distorted_img = cv2.imread(path_of_images+"/0.jpg")
h,  w = distorted_img.shape[:2]
new_camera_mtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv2.undistort(distorted_img, mtx, dist, None, new_camera_mtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(path_of_images +"/calibration_result_0.jpg",dst)


#***********************************************Pose Estimation******************************************************
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, edge_of_square, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
count = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _ ,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img_wpose = draw(img,corners2,imgpts)
        cv2.imshow('img',img_wpose)
        cv2.imwrite(path_of_images +"/wpose_"+str(count)+".jpg", img_wpose)
        cv2.waitKey(500)
        count+=1
print("images with poses are saved")    
cv2.destroyAllWindows()


