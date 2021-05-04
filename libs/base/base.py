import abc 
import cv2 
import numpy as np 
import functools
from ..base.network import nn


class base(metaclass=abc.ABCMeta):
    def __init__(self, det: str, reg: str, device: str, color:tuple):
        self.detector = nn(det, device)
        self.recog = nn(reg, device)
        self.curr_id = 0
        self.next_id = 1
        self.color = color

    def run(self, image):
        raise NotImplemented()
    
    def recog(self, image):
        raise NotImplemented()

    def draw_box(self, image, box):
        (xmin, ymin, xmax, ymax) = box 
        cv2.rectangle(image, (xmin, ymin - 22),
                      (xmax, ymin), self.color, -1)
        cv2.rectangle(image, (xmin, ymin - 22),
                      (xmax, ymin), (255, 255, 255))
        cv2.rectangle(image, (xmin, ymin),
                      (xmax, ymax), self.color, 1)
        

    def get_detections(self, detections, image):
        frame_h, frame_w = image.shape[:2]
        det_id = []
        boxes = []
        for face_id, face in enumerate(detections):
            box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            (xmin, ymin, xmax, ymax) = box.astype("int")
            boxes.append((xmin, ymin, xmax, ymax))
            face_frame = image[ymin:ymax, xmin:xmax]
            det_id.append(face_frame)

        return det_id, boxes

    def run_async(self, image):
        detections = self.detector.async_infer(image)
        if detections:
            framedet_idxss, boxes = self.get_detections(detections[0][0], image)

    @abc.abstractmethod
    def run_video(self, vid, callback:callable= None, cv2_display=True):
        cap = cv2.VideoCapture(vid)
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break 
            _frame = self.infer_video(frame)
            if cv2_display:
                cv2.imshow(self.__class__.__name__, _frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if callback:
                callback(_frame)
