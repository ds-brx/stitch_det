import cv2
import numpy as np
import os
import time

classFile = 'coco.names'
classNames = []

with open (classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
# for i in net.getUnconnectedOutLayers():
#     print(i)
ln = [ln[199], ln[226], ln[253]]
# print(ln)

def detect (img, count):
    (H, W) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    
    net.setInput(blob)
    
    start = time.time()
    
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                        0.3)

                if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        # color = [int(c) for c in classNames[classIDs[i]]]
                        cv2.rectangle(img, (x, y), (x + w, y + h),(255, 0, 0), 2)
                        text = "{}: {:.4f}".format(classNames[classIDs[i]], confidences[i])
                        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(255, 0, 0), 2)

                        
                    cv2.imwrite('det_output/img_file_{}.jpeg'.format(count), img)
                    # cv2.imshow('img_file',img)
                    # cv2.waitKey(0)
    return img
                    