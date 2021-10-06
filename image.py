import cv2
import os
import sys

background = None
BG_WEIGHT = 0.5

cam = cv2.VideoCapture(0)

start = False
counter = 0
num_samples = 200
IMG_SAVE_PATH = 'test_img'

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

region_top = 0
region_bottom = FRAME_HEIGHT * 2 // 3
region_left = FRAME_WIDTH // 2
region_right = FRAME_WIDTH

def get_region(img):
    # Separate the region of interest from the rest of the frame.
    region = img[region_top:region_bottom, region_left:region_right]
    # Make it grayscale so we can detect the edges more easily.
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Use a Gaussian blur to prevent frame noise from being labeled as an edge.
    region = cv2.GaussianBlur(region, (5,5), 0)

    return region

def get_average(region):
    # We have to use the global keyword because we want to edit the global variable.
    global background
    # If we haven't captured the background yet, make the current region the background.
    if background is None:
        background = region.copy().astype("float")
        return
    # Otherwise, add this captured frame to the average of the backgrounds.
    cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment(image, threshold=25):
    global background
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass

num_frames = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    if counter == num_samples:
       break

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (0, 255, 0), 2)

    k = cv2.waitKey(1)
    if k == ord('o'):
            name = 'okay'
            IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, name)
            try:
                os.mkdir(IMG_CLASS_PATH)
            except FileExistsError:
                os.remove(IMG_CLASS_PATH)
                os.mkdir(IMG_CLASS_PATH)
            
                  
    if k == ord('v'):
            name = 'v_sign'
            IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, name)
            try:
                os.mkdir(IMG_CLASS_PATH)
            except FileExistsError:
                os.remove(IMG_CLASS_PATH)
                os.mkdir(IMG_CLASS_PATH)
            
    if k == ord('t'):
            name = 'thumb_up'
            IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, name)
            try:
                os.mkdir(IMG_CLASS_PATH)
            except FileExistsError:
                os.remove(IMG_CLASS_PATH)
                os.mkdir(IMG_CLASS_PATH)

    roi = get_region(frame)
    if num_frames < 30:
        get_average(roi)
    else:
        hand = segment(roi)
    
        if start:
            if hand is not None:
                (thresholded, segmented) = hand

                save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(counter + 1))
                print(save_path)
                cv2.imwrite(save_path, thresholded)
                counter += 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,"Collecting {}".format(counter),
            (10, 20), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    num_frames += 1


    if k == ord('a'):
        start = not start

    if k == ord('q'):
            break

print("\n{} image(s) saved to {}".format(counter, IMG_CLASS_PATH))
cam.release()
cv2.destroyAllWindows()