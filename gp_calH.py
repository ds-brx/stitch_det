# HOMOGRAPHY MATRICES AND STORE IN FILES



from stitch_calH import stitch_image
import cv2
import argparse
import time
import numpy as np

leftMost    = cv2.VideoCapture('1.mp4')
leftCenter  = cv2.VideoCapture('2.mp4')
rightCenter = cv2.VideoCapture('3.mp4')    
rightMost   = cv2.VideoCapture('4.mp4')

## Can we alter dimensions of width and height? Make it smaller for easier detection?

# leftMost    = cv2.VideoCapture(3,cv2.CAP_DSHOW)
# leftMost.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# leftMost.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# # leftMost.set(15, -6.0)

# leftCenter    = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# leftCenter.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# leftCenter.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# # leftCenter.set(15, -4.0)

# rightCenter = cv2.VideoCapture(1,cv2.CAP_DSHOW)
# rightCenter.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# rightCenter.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# rightMost = cv2.VideoCapture(2,cv2.CAP_DSHOW)
# rightMost.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# rightMost.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

alpha = np.load('alpha.npy')
beta = np.load('beta.npy')
def main(key_frame_num): # Number of input images to be stitched as one
    temp = np.load('nxx.npy') # Cylindrical Coordinate warping Matrix
    frame_num = 1
    iteration = 1
    while True:

        if iteration == 1:
            time.sleep(10)
        iteration += 1
        ret1,l1 = leftMost.read()    # Maybe just use underscores inplace of ret
        ret2,l2 = leftCenter.read()
        ret3,l3 = rightCenter.read()
        ret4,l4 = rightMost.read()
        ims = [l1,l2,l3,l4]
        current_img = l1
        tic = time.time()
        for i in range(1,4):
            next_img = ims[i]
            #print("Stitching frame{} and frame{}...".format(i - 1, i))
            current_img = stitch_image(current_img, next_img, frame_num, temp, i) 
            # For next iteration current stitched image is passed as left part
            # & next_img is .....right part !
            frame_num = 0
        cv2.imshow('output',current_img)
        toc = time.time()
        print("Fps :",1/((toc-tic)))
        time.sleep(3)
        #cv2.imwrite('output.jpg', stitched)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num', nargs='?', default=4, type=int,
                        help="the number of key frames (default: 4)")
    args = parser.parse_args()

    main(args.num)
