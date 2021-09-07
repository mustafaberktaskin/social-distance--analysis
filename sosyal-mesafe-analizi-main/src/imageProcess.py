import cv2
import numpy as np

class imageProcess:
    def __init__(self,frame,darkNetModel,layers,labels):
        self.frame = frame
        (self.frameHeight,self.frameWidth) = self.frame.shape[:2]
        self.darkNetModel = darkNetModel
        self.layers = layers
        self.labels = labels
        self.scoreThreshold = 0.5
        self.NMSThreshold = 0.3
        self.confidenceThreshold = 0.5
        self.confidences = []
        self.bBoxes = []

    def getLayerOutputs(self):
        blob = cv2.cv2.dnn.blobFromImage(self.frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.darkNetModel.setInput(blob)
        layerOutputs = self.darkNetModel.forward(self.layers)
        return layerOutputs

    def setBoundingBoxesBeforeNMS(self,layerOutputs):
        for layerOutput in layerOutputs:
            for detection in layerOutput:
                scores = detection[5:]
                labelID = np.argmax(scores)
                confidence = scores[labelID]
                if self.labels[labelID] == "person" and confidence > self.confidenceThreshold:
                    box = detection[0:4] * np.array([self.frameWidth, self.frameHeight, self.frameWidth, self.frameHeight])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    self.bBoxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(confidence))
    
    def getBoundingBoxesByNMS(self):
        bBoxesAfterNMS = cv2.cv2.dnn.NMSBoxes(self.bBoxes,self.confidences,self.scoreThreshold,self.NMSThreshold)
        return bBoxesAfterNMS
    
    def drawRectangle(self,person):
        if person.status == True:
            cv2.cv2.rectangle(self.frame, (person.x, person.y), (person.x + person.width, person.y + person.heigth), (0, 0, 150), 2)
        if person.status == False:
            cv2.cv2.rectangle(self.frame, (person.x, person.y), (person.x + person.width, person.y + person.heigth), (0, 255, 0), 2)
    
    def putText(self,personPairs):
        index = 2
        for personPair in personPairs:
            personIndex = 0 if personPair[0].x <= personPair[1].x else 1
            if index % 2 == 1:
                cv2.cv2.putText(self.frame,"Sosyal Mesafe Ihlali",(personPair[personIndex].x-7,personPair[personIndex].centroid[1]),cv2.cv2.FONT_HERSHEY_SIMPLEX,0.3,(0, 0, 255),1)
            index += 1

    def getFinalFrame(self):
        return self.frame.copy()

    def getBoundingBoxesBeforeNMS(self):
        return self.bBoxes






        