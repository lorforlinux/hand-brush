from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
from collections import deque


# For Drawing toolbox!

# Setup deques to store separate colors in separate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

# Fist Or hand
fist_thresh = 1.2

# shift the toolkit 
x_shift = 300
y_shift = 50

# color palete
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
colorIndex = 0



# load the model
detection_graph, sess = detector_utils.load_inference_graph()

score_thresh = 0.4
fps = 1

# 0 for internal web cam
video_source = 0

# caputer frame
cap = cv2.VideoCapture(video_source)

start_time = datetime.datetime.now()
num_frames = 0

# get dimenstions
im_width, im_height = (cap.get(3), cap.get(4))

# max number of hands we want to detect/track
num_hands_detect = 1

cv2.namedWindow('Hand Brush', cv2.WINDOW_NORMAL)

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

    # draw bounding boxes
    # detector_utils.draw_box_on_image(
    #     num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)


    # Add the coloring options to the frame
    image_np = cv2.rectangle(image_np, (x_shift+40,y_shift+1), (x_shift+140,y_shift+65), (122,122,122), -1)
    image_np = cv2.rectangle(image_np, (x_shift+160,y_shift+1), (x_shift+255,y_shift+65), colors[0], -1)
    image_np = cv2.rectangle(image_np, (x_shift+275,y_shift+1), (x_shift+370,y_shift+65), colors[1], -1)
    image_np = cv2.rectangle(image_np, (x_shift+390,y_shift+1), (x_shift+485,y_shift+65), colors[2], -1)
    image_np = cv2.rectangle(image_np, (x_shift+505,y_shift+1), (x_shift+600,y_shift+65), colors[3], -1)
    cv2.putText(image_np, "CLEAR ALL", (x_shift+49, y_shift+33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_np, "BLUE", (x_shift+185,y_shift+ 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_np, "GREEN", (x_shift+298,y_shift+ 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_np, "RED", (x_shift+420,y_shift+ 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_np, "YELLOW", (x_shift+520,y_shift+ 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)


    for i in range(num_hands_detect):
        if(scores[i] > score_thresh):
            # calculate centroid
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,boxes[i][0] * im_height, boxes[i][2] * im_height)
            # calculate centroid
            y = (bottom +top)/2
            x = (right + left)/2
            
            # a fist OR exapnded hand
            fist = (top-bottom)/(left-right)
            # use y insted of top to get the pointer in center of hand (on plam)
            center = (int(x), int(top))
            cv2.circle(image_np,  center, 5, colors[colorIndex], 10)

            # if it's a straight hand and not a fist or open hand
            if fist>fist_thresh:
                if center[1] <= y_shift+65:
                    if x_shift+40 <= center[0] <= x_shift+140: # Clear All
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        bindex = 0
                        gindex = 0
                        rindex = 0
                        yindex = 0

                    elif x_shift+160 <= center[0] <= x_shift+255:
                            colorIndex = 0 # Blue
                    elif x_shift+275 <= center[0] <= x_shift+370:
                            colorIndex = 1 # Green
                    elif x_shift+390 <= center[0] <= x_shift+485:
                            colorIndex = 2 # Red
                    elif x_shift+505 <= center[0] <= x_shift+600:
                            colorIndex = 3 # Yellow
                else :
                    if colorIndex == 0:
                        bpoints[bindex].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[gindex].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[rindex].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yindex].appendleft(center)

            # if it is not a fist (OR expanded hand)
            else:
                bpoints.append(deque(maxlen=512))
                bindex += 1
                gpoints.append(deque(maxlen=512))
                gindex += 1
                rpoints.append(deque(maxlen=512))
                rindex += 1
                ypoints.append(deque(maxlen=512))
                yindex += 1
            

        else:
            bpoints.append(deque(maxlen=512))
            bindex += 1
            gpoints.append(deque(maxlen=512))
            gindex += 1
            rpoints.append(deque(maxlen=512))
            rindex += 1
            ypoints.append(deque(maxlen=512))
            yindex += 1
                
    # Draw lines of all the colors (Blue, Green, Red and Yellow)
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(image_np, points[i][j][k - 1], points[i][j][k], colors[i], 2)


    # Calculate Frames per second (FPS)
    num_frames += 1
    elapsed_time = (datetime.datetime.now() -
                    start_time).total_seconds()
    fps = num_frames / elapsed_time

    
    # Display FPS on frame
    if (fps > 0):
        detector_utils.draw_fps_on_image(
            "FPS : " + str(int(fps)), image_np)

    cv2.imshow('Hand Brush', cv2.cvtColor(
        image_np, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
    