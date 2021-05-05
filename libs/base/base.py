import abc 
import cv2 
import numpy as np 
import functools
from ..base.network import nn


class base(metaclass=abc.ABCMeta):
    def __init__(self, det: str, reg: str, device: str, color:tuple):
        self.detector = nn(det, device)
        self.recognizer = nn(reg, device)
        self.curr_id = 0
        self.next_id = 1
        self.color = color
    @abc.abstractmethod
    def run(self, image):
        pass #raise NotImplemented()
    @abc.abstractmethod
    def recog(self, image):
        pass #raise NotImplemented()
    @abc.abstractmethod
    def draw_recog(self, image, recog_res):
        pass #raise NotImplemented()
    @abc.abstractmethod
    def infer_video(self, frame):
        pass 

    def draw_box(self, image, box):
        xmin, ymin, xmax, ymax = [int(each) for each in list(box)]
        cv2.rectangle(image, (xmin, ymin ),(xmax, ymin), self.color, 1)
        cv2.rectangle(image, (xmin, ymin ),\
                      (xmax, ymin), (255, 255, 255))
        cv2.rectangle(image, (xmin, ymin),\
                      (xmax, ymax), self.color, 1)
        
    def get_detections(self, detections, image):
        """Get detections

        Args:
            detections (numpy.ndarray): numpy detection array
            image (numpy.ndarray): image

        Returns:
            list(conf, id, box): list of (confidence, id, box)
        """
        frame_h, frame_w = image.shape[:2]
        det_id = []
        boxes = []
        conf = []
        for face_id, detection in enumerate(detections):
            conf.append(float(detection[2]))
            box = detection[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            (xmin, ymin, xmax, ymax) = box.astype("int")
            boxes.append((xmin, ymin, xmax, ymax))
            face_frame = image[ymin:ymax, xmin:xmax]
            det_id.append(face_frame)

        return conf, det_id, boxes

    def run_async(self, image, det_threshold=0.5):
        detections = self.detector.async_infer(image)[self.detector.out_name]
        if not detections is None:
            conf, framedet_idxss, boxes = self.get_detections(detections[0][0], image)
            for i, (frame_id, box) in enumerate(zip(framedet_idxss, boxes)):
                if conf[i] >= det_threshold:
                    res = self.recog(frame_id)
                    self.draw_box(image, box)
                    self.draw_recog(image, res)


    def run_video(self, vid, callback:callable= None, cv2_display=True):
        cap = cv2.VideoCapture(vid)
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break 
            _frame = self.infer_video(frame)
            if cv2_display:
                cv2.imshow(self.__class__.__name__, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if callback:
                callback(_frame)
