from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

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

    # draw bounding boxes
    detector_utils.draw_box_on_image(
        num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

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
    