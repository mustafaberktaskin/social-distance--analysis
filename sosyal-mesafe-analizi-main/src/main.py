from yolo import yolo
from imageProcess import imageProcess
from videoWriter import videoWriter
from utils import *
import os
import cv2
import imutils
import time

inputVideoPath = os.path.sep = "../input-video/test.mp4"


videoWriterObj = videoWriter()
yoloObj = yolo()
yoloLabels = yoloObj.getLabels()
yoloDarkNetModel = yoloObj.getDarkNetModel()
yoloLayers = yoloObj.getLayers(yoloDarkNetModel)


video = cv2.cv2.VideoCapture(inputVideoPath)
frameCounter = 0
totalFrame = int(video.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
time1 = time.time()
flag = False
while video.isOpened():
    check, frame =  video.read()
    if not check:
        break
    currentFrame = frame.copy()
    currentFrame = imutils.resize(currentFrame, width=480)
    print("Remaining Frame : {}".format(totalFrame-frameCounter))
    frameCounter += 1

    imageProcessObj = imageProcess(currentFrame,yoloDarkNetModel,yoloLayers,yoloLabels)
    layerOutputs = imageProcessObj.getLayerOutputs()
    imageProcessObj.setBoundingBoxesBeforeNMS(layerOutputs)
    NMSBoundingBoxes = imageProcessObj.getBoundingBoxesByNMS()
    BoudingBoxes = imageProcessObj.getBoundingBoxesBeforeNMS()

    peopleList = getPeopleFeaturesList(NMSBoundingBoxes,BoudingBoxes)

    personPairs = []
    for i in range(len(peopleList)):
        for j in range(len(peopleList)):
            width = calculatePersonWidth(peopleList[i],peopleList[j])
            distanceStatus = checkDistanceProximityStatus(peopleList[i],peopleList[j],width)

            if distanceStatus:
                personPairs.append([peopleList[i],peopleList[j]])
                peopleList[i].status = True
                peopleList[j].status = True

    for p in peopleList:
        imageProcessObj.drawRectangle(p)
    
    imageProcessObj.putText(personPairs)
    finalFrame = imageProcessObj.getFinalFrame()
    
    #cv2.cv2.imshow('output', finalFrame)
    videoWriterObj.write(finalFrame,flag)
    flag = True
    del imageProcessObj

    if cv2.cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
time2 = time.time()
print("Total Time: {}".format((time2-time1)/60))
video.release()
cv2.cv2.destroyAllWindows()


    
    



