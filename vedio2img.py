import tensorflow as tf
import numpy as np
import scipy.misc as misc
from PIL import Image
import cv2

def capture_vedio():
    cap = cv2.VideoCapture("movie.mkv")
    if cap.isOpened() == False:
        print("Error opening vedio stream or file")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = np.array(frame)
            if count % 24 == 0:
                Image.fromarray(frame).save("E://DeepLearn_Experiment//CartoonSet//"+str(count)+".jpg")
            if count % (48*60) == 0:
                print("Number %d minutes"%(count//(24*60)))
            count += 1
            # cv2.imshow("Frame", np.uint8((frame)))
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_vedio()

