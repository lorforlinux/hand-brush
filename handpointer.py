from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np 
from collections import deque

pts = deque(maxlen = 64)


# load the model
detection_graph, sess = detector_utils.load_inference_graph()


score_thresh = 0.4
fps = 1

# 0 for internal web cam
video_source = 0

# caputer frame
cap = cv2.VideoCapture(0)

start_time = datetime.datetime.now()
num_frames = 0

# get dimenstions
im_width, im_height = (cap.get(3), cap.get(4))

# max number of hands we want to detect/track
num_hands_detect = 1

cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)

while True:
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    ret, image_np = cap.read()
    image_np = cv2.flip(image_np, 1)
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")

    # actual detection
    boxes, scores = detector_utils.detect_objects(
        image_np, detection_graph, sess)

    # Draw Pointer trails
    
    # If we detect hand
    if(scores[0] > score_thresh):
        # calculate centroid
        (left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width,boxes[0][0] * im_height, boxes[0][2] * im_height)
        y = (bottom +top)/2
        x = (right + left)/2
        print("detected")
        # a fist OR exapnded hand
        fist = (top-bottom)/(left-right)
        center = (int(x), int(y))
        cv2.circle(image_np,  center, 5, (255, 25, 0), 10)

        # if it is not a fist OR expanded hand
        if fist>1.2:
            pts.appendleft(center)
        else:
            pts.appendleft(None)

    else:
        pts.appendleft(None)        
    

    # Draw trails
    for i in range (1,len(pts)):
        if pts[i-1]is None or pts[i] is None:
            continue
        thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
        cv2.line(image_np, pts[i-1],pts[i],(255,0,0),thick)

    # Calculate Frames per second (FPS)
    num_frames += 1
    elapsed_time = (datetime.datetime.now() -
                    start_time).total_seconds()
    fps = num_frames / elapsed_time

    
    # Display FPS on frame
    if (fps > 0):
        detector_utils.draw_fps_on_image(
            "FPS : " + str(int(fps)), image_np)

    cv2.imshow('Hand Detection', cv2.cvtColor(
        image_np, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
    