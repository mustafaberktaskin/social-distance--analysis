import os
import cv2

class yolo:
    def __init__(self):
        self.weigths = os.path.sep = "../yolo/yolov3.weights"
        self.config = os.path.sep = "../yolo/yolov3.cfg"
        self.labelsPath = os.path.sep = "../yolo/coco.names"
    
    def getLabels(self):
        labels = open(self.labelsPath).read().strip().split("\n")
        return labels
    
    def getDarkNetModel(self):
        darkNetModel = cv2.cv2.dnn.readNetFromDarknet(self.config,self.weigths)
        return darkNetModel
    
    def getLayers(self,darkNetModel):
        layers = darkNetModel.getLayerNames()
        layers = [layers[i[0] - 1] for i in darkNetModel.getUnconnectedOutLayers()]
        return layers