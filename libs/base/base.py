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
        self._rectangle_thickness = 2

    @property
    def rect_thickness(self):
        return self._rectangle_thickness
    @rect_thickness.setter
    def rect_thickness(self, value):
        self._rectangle_thickness = value


    @abc.abstractmethod
    def run(self, image):
        pass #raise NotImplemented()
    @abc.abstractmethod
    def recog(self, image):
        pass #raise NotImplemented()
    @abc.abstractmethod
    def draw_recog(self, image, box, recog_res, **kwargs):
        pass #raise NotImplemented()
    @abc.abstractmethod
    def infer_video(self, frame):
        pass 

    def draw_box(self, image, box):
        xmin, ymin, xmax, ymax = [int(each) for each in list(box)]
        cv2.rectangle(image, (xmin, ymin ),(xmax, ymin), self.color, 2)
        cv2.rectangle(image, (xmin, ymin ),\
                      (xmax, ymin), (255, 255, 255))
        cv2.rectangle(image, (xmin, ymin),\
                      (xmax, ymax), self.color, self._rectangle_thickness)
        

    def detect(self, image):
        """Detect object

        Args:
            image (numpy.ndarray): image

        Returns:
            list: list(conf, id, box)
        """
        detections = self.detector.async_infer(image)[self.detector.out_name].buffer[0][0]
        if not detections is None:
            frame_h, frame_w = image.shape[:2]
            det_id = []
            boxes = []
            conf = []
            for _id, detection in enumerate(detections):
                box = detection[3:7] * \
                    np.array([frame_w, frame_h, frame_w, frame_h])
                (xmin, ymin, xmax, ymax) = box.astype("int")
                face_frame = image[ymin:ymax, xmin:xmax]
                if any(c == 0 for c in face_frame.shape):
                    continue
                boxes.append((xmin, ymin, xmax, ymax))
                conf.append(float(detection[2]))
                det_id.append(face_frame)
            return conf, det_id, boxes
        else:
            return None

    def run_async(self, image, det_threshold=0.5, draw= True, callback:callable = None):
        """Async inference

        Args:
            image (numpy.ndarray): image
            det_threshold (float, optional): detection threshold. Defaults to 0.5.
            draw(bool, optional): draw detection boxes. Defaults to True
            callback(callable, optional): func handles @recog method result
        Returns:
            dict: {'det': list(box(xmin, ymin, xmax, ymax)), 'rec': list(result of 'recog' method) }
        """
        detections = self.detect(image)
        if detections:
            result = {'det': [], 'rec': []}
            conf, framedet_idxss, boxes = detections
            for i, (frame_id, box) in enumerate(zip(framedet_idxss, boxes)):
                if conf[i] >= det_threshold:
                    res = self.recog(frame_id)
                    if draw:
                        self.draw_box(image, box)
                    if callback:
                        callback(image, box, res)
                    result['det'].append(box)
                    result['rec'].append(res)
            return result
        else:
            return None

    def run_video(self, vid, callback:callable= None, cv2_display=True):
        cap = cv2.VideoCapture(vid)
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                cv2.destroyAllWindows()
                break
            res = self.infer_video(frame)
            if callback:
                callback(res)
            if cv2_display:
                cv2.namedWindow(self.__class__.__name__, cv2.WINDOW_KEEPRATIO)
                cv2.imshow(self.__class__.__name__, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
