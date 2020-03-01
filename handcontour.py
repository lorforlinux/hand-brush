from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np

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
num_hands_detect = 2
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

    # Draw contours
    for i in range(num_hands_detect):
        # If we detect hand
        if(scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,boxes[i][0] * im_height, boxes[i][2] * im_height)
            # create a black frame
            black = np.zeros((int(im_height), int(im_width), 3), np.uint8) 
            # make the detected area white
            black1 = cv2.rectangle(black,(int(left),int(top)),(int(right),int(bottom)),(255, 255, 255), -1)
            gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY) 
            ret,b_mask = cv2.threshold(gray,127,255, 0) 
            # get the hand data
            fin = cv2.bitwise_and(black1, image_np, mask = b_mask)
            output = cv2.Canny(fin, 100, 200)
            # get the contours
            image, contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # draw contours
            image = cv2.drawContours(image_np, contours, -1, (0, 255, 0), 2)


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
    