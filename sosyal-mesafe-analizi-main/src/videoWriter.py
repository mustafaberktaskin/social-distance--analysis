import cv2

class videoWriter:
    def __init__(self):
        self.fourcc = cv2.cv2.VideoWriter_fourcc(*'mp4v')
        self.outputPath = "../output-video/output-video.mp4"
        self.create = None
    
    def write(self,frame,flag):
        if flag == False:
            self.create = cv2.cv2.VideoWriter(self.outputPath,self.fourcc,30,(frame.shape[1], frame.shape[0]), True)
        self.create.write(frame)



