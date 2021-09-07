import math
from person import person

def checkDistanceProximityStatus(personX,personY,width):
    distance = math.sqrt((personY.centroid[1]-personX.centroid[1])**2 + (personY.centroid[0]-personX.centroid[0])**2)
    return 0 < distance < width 


def calculatePersonWidth(personX,personY):
    return personX.width if personX.width > personY.width else personY.width

def getPeopleFeaturesList(NMSBBoxes,BoudingBoxes):
        flattenBoxes = NMSBBoxes.flatten()
        peopleList = []
        for i in flattenBoxes:
            personObj = person()
            (personObj.x, personObj.y) = (BoudingBoxes[i][0], BoudingBoxes[i][1])
            (personObj.width, personObj.heigth) = (BoudingBoxes[i][2], BoudingBoxes[i][3])
            personObj.centroid = [int(personObj.x + personObj.width / 2), int(personObj.y + personObj.heigth / 2)]
            personObj.status = False
            peopleList.append(personObj)
        return peopleList

