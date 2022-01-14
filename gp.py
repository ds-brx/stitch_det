# STICHING 



from stitch import stitch_image
import cv2
import argparse
import time
import numpy as np
from detection import detect

leftMost    = cv2.VideoCapture('1.mp4')
leftCenter  = cv2.VideoCapture('2.mp4')
rightCenter = cv2.VideoCapture('3.mp4')    
rightMost   = cv2.VideoCapture('4.mp4')

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

# # Get width and height of video stream
# w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
# h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Get frames per second (fps)
# fps = stream.get(cv2.CAP_PROP_FPS)
 
# # Define the codec for output video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
 
# # Set up output video
# outputVideo = cv2.VideoWriter('video2_FAST.mp4', fourcc, fps, (w, h))

# ## ============================== DETECTION PREP ================================== ##
# # COPIED FROM : https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

# classFile = 'coco.names'
# classNames = []

# with open (classFile, 'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

# weightsPath = "yolov3.weights"
# configPath = "yolov3.cfg"

# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# ln = net.getLayerNames()
# # for i in net.getUnconnectedOutLayers():
# #     print(i)
# ln = [ln[199], ln[226], ln[253]]
# # print(ln)

# ## ============================== DETECTION PREP ================================== ##

alpha = np.load('alpha.npy')   #what is this?
beta = np.load('beta.npy')
def main(key_frame_num): # Number of input images to be stitched as one

    '''
    warping matrix pipeline?
    '''

    temp = np.load('nxx.npy') # Cylindrical Coordinate warping Matrix

    frame_num = 1
    count = 0
    while True:
        count += 1
        ret1,l1 = leftMost.read()    # Maybe just use underscores inplace of ret
        ret2,l2 = leftCenter.read()
        ret3,l3 = rightCenter.read()
        ret4,l4 = rightMost.read()
        ims = [l1,l2,l3,l4]
        current_img = l1
        tic = time.time()

        '''
         parallelize - multi thread/ multi thread cuda
         2 at a time
         change the loop format

        '''
        
        for i in range(1,4):
            next_img = ims[i]
            #print("Stitching frame{} and frame{}...".format(i - 1, i))
            current_img = stitch_image(current_img, next_img, frame_num, temp, i) 
            # For next iteration current stitched image is passed as left part
            # & next_img is .....right part !
            frame_num = 0
        # cv2.imwrite('output/output.jpeg',current_img)
        # current_img = cv2.imread('output/output.jpeg')
        current_img = cv2.cvtColor(current_img,cv2.COLOR_BGRA2BGR)
        current_img = detect(current_img, count)
        # cv2.imwrite('stiched/{}.jpeg'.format(count), current_img)

        toc = time.time()
        print("Fps :",1/((toc-tic)))
        # cv2.imwrite('det_output/{}.jpeg'.format(count), current_img)
        cv2.imshow('output',current_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num', nargs='?', default=4, type=int,
                        help="the number of key frames (default: 4)")
    args = parser.parse_args()

    main(args.num)
