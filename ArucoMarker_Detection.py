import numpy as np
import cv2 

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
# detector parameters: https://docs.opencv.org/3.4/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
arucoParams = cv2.aruco.DetectorParameters_create()
markerSize = 0.125 #meters
# camera matrix and distortion coefficients results from camera calibration
mtx =np.array( [ [8.6297319278294219e+02,          0.          , 3.2410970150269952e+02], 
                 [           0.         ,8.6289007075149686e+02, 2.3151809145852621e+02],
                 [           0.         ,          0.          ,            1.         ] ])
dist =np.array( [ 1.3876171595109568e-01, -5.0495233579131149e-01, -1.4364550355797534e-03, 3.0938437583063767e-03, 1.3034844698951493e+00 ])

#Capturing Real Time Video
cap = cv2.VideoCapture(0)
#width
#cap.set(3,432)
#height
#cap.set(4,432)
prev_frame_time = 0
new_frame_time = 0
if cap.isOpened():
    ret , frame = cap.read()
    #cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
    #ret will store that bool value 
else:
    ret = False

while ret:
    ret , frame = cap.read()
    # Calculating the fps
    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    cv2.putText(frame,"fps: "+str(int(fps)),(15,15), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,0,255),1,cv2.LINE_AA) # displays fps
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # apply grayscale
    blur = cv2.GaussianBlur(gray,(5,5),0) # blured (filtered) image with a 5x5 gaussian kernel to remove the noise
    # Draw detected markers in image.
    # image =   cv.aruco.drawDetectedMarkers(   image, corners[, ids[, borderColor]]    )
    # cv2.aruco.detectMarkers(  image, dictionary[, corners[, ids[, parameters[, rejectedImgPoints[, cameraMatrix[, distCoeff]]]]]] )
    (corners, ids, rejected) = cv2.aruco.detectMarkers(blur, arucoDict, parameters= arucoParams,cameraMatrix= mtx, distCoeff= dist)

    if(len(corners) > 0): # Returns True if at least one ArUco marker is detected.
        ids = ids.flatten() # Return a copy of the ids array collapsed into one dimension.
        try:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor = (0, 0, 255) )  # Draw A square around the markers
        except cv2.error as e:
            print(e)
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # Pose estimation for single markers.
            # rvecs, tvecs, _objPoints  =   cv.aruco.estimatePoseSingleMarkers( corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs[, _objPoints]]] )
            # markerLength  the length of the markers' side. The returning translation vectors will be in the same unit. Normally, unit is meters.
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, markerSize, mtx, dist)
            #(rvec - tvec).any()  # get rid of that nasty numpy value array error

            # image =   cv.aruco.drawAxis(  image, cameraMatrix, distCoeffs, rvec, tvec, length )
            cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, (0.5)*markerSize)  # Draw the x,y,z axes

            # extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            # Compute and draw the center (x, y) coordinates of the detected ArUco marker.
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 15, (0, 0, 255), 1)

            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),(topLeft[0] +15 , topLeft[1] - 15),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

    cv2.imshow("Real-time frame", frame)
    if cv2.waitKey(1) == 27:   #press esc to quit
        break
cv2.destroyAllWindows()
cap.release()
