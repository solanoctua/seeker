import cv2 
import os 
  
# Get the current working directory 
cwd = os.getcwd() 
#path_of_images = "C:/Users/..../Desktop/"
path_of_images = cwd
cap = cv2.VideoCapture(0)
#frame_width  = cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)   
#frame_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)  
count = 0
if cap.isOpened():
    ret,frame = cap.read()
else:
    ret = False
while ret:
    ret , frame = cap.read()
    cv2.putText(frame,"press 'esc' to quit",(15,15), cv2.FONT_HERSHEY_SIMPLEX, .4,(0,0,255),1,cv2.LINE_AA) #displays some text
    cv2.putText(frame,"press 's' to save the image ({})".format(cwd),(15,35), cv2.FONT_HERSHEY_SIMPLEX, .4,(255,0,0),1,cv2.LINE_AA) #displays some text 
    cv2.imshow("camera",frame)
    
    key = cv2.waitKey(1)
    if key == ord("s"):
        print("Saving image to ",path_of_images)
        cv2.imwrite("{}{}.jpg".format(path_of_images+"/",count), frame)
        count += 1
    if key == 27:   #press esc to quit
        break
cv2.destroyAllWindows()
cap.release()
