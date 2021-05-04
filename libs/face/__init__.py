import numpy as np 

from ..base.network import nn 
from ..base.base import base

class face_age_gender(base):

    def __init__(self, det:str, reg:str, device:str):
        super(face_age_gender, self).__init__(det, reg, device)
        

    def run(self, image):
        res = self.detector.sync_infer(image)

    def get_detections(self, detections, image):
        frame_h, frame_w = image.shape[:2]
        face_frames = []
        boxes = []

        for face_id, face in enumerate(detections):
            box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            (xmin, ymin, xmax, ymax) = box.astype("int")
            boxes.append((xmin, ymin, xmax, ymax))
            face_frame = image[ymin:ymax, xmin:xmax]
            #face_frame = resize_frame(face_frame, resize_width)
            face_frames.append(face_frame)

        return face_frames, boxes

        
        

    
