import time 
import numpy as np
import cv2
from scipy.spatial import distance
from munkres import Munkres
from ..base.network import nn
from ..base.base import base


class tracking_object:
    def __init__(self, pos, feature, id=-1):
        self.feature = feature
        self.id = id
        self.time = time.monotonic()
        self.pos = pos
        self.catchup = 0
        self.curr_showup = True
        self.inactive = -1
class person_reid(base):

    def __init__(self, det: str, reg: str, device: str, color: tuple, # base
            dist_threshold=3., timeout_threshold= 14):
        super(person_reid, self).__init__(det, reg, device, color)
        self.id_num = 0
        self.dist_threshold = dist_threshold
        # Tracking Object database timeout (sec)
        self.timeout_threshold = timeout_threshold
        # Tracking Object database (feature, id)
        self.tracking_objects = []         
        self.new_objects = []
        self.catchup = 5

    def recog(self, image):
        output = self.recognizer.async_infer(image)[self.recognizer.out_name]
        return np.array(output.buffer.reshape(256))

    def draw_recog(self, image, box, recog_res):
        age, gender = recog_res
        xmin, ymin, xmax, ymax = [int(each) for each in list(box)]
        text = f'{self.gender[gender]}age {str(age)}'
        cv2.putText(image, text, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.color, 1, cv2.LINE_AA)
        return

    def run(self, image, **kwargs):
        """
        Args:
            image(numpy.ndarray): image
            **kwargs: 
                draw(bool, optional): draw result on image, Default is True
        """
        draw = kwargs.get('draw', True)
        res = self.run_async(image, det_threshold=0.6, draw= draw, )
        # Create cosine distance matrix and match objects in the frame and the DB
        if len(res['det']) == 0:
            return None
        objects = []
        for vec, box in zip(res['rec'], res['det']):
            objects.append(tracking_object(box, vec))
        hangarian = Munkres()
        dist_matrix = [[distance.cosine(obj_db.feature, obj_cam.feature)
                        for obj_db in self.tracking_objects] for obj_cam in objects]
        combination = hangarian.compute(dist_matrix)# Solve matching problem
        for idx_obj, idx_db in combination:
            # This object has already been assigned an ID
            if objects[idx_obj].id != -1:
                continue
            dist = distance.cosine(
                objects[idx_obj].feature, self.tracking_objects[idx_db].feature)
            if dist < self.dist_threshold:
                self.tracking_objects[idx_db].time = time.monotonic()
                self.tracking_objects[idx_db].pos = objects[idx_obj].pos
                self.tracking_objects[idx_db].feature = objects[idx_obj].feature
                objects[idx_obj].id = self.tracking_objects[idx_db].id
        """
        # Remove noise
        del hangarian
        if self.new_objects == []:
            for obj in objects:
                if obj.id == -1:
                    self.new_objects.append(obj)
        else:
            new_objects = []
            for obj in objects:
                if obj.id == -1:
                    new_objects.append(obj)
            new_hangarian = Munkres()
            dist_matrix = [[distance.cosine(obj_db.feature, obj_cam.feature)
                            for obj_db in self.new_objects] for obj_cam in new_objects]
            if dist_matrix:
                combination = new_hangarian.compute(dist_matrix)  # Solve matching problem
                for idx_obj, idx_db in combination:
                    dist = distance.cosine(
                        new_objects[idx_obj].feature, self.new_objects[idx_db].feature)
                    if dist < self.dist_threshold:
                        self.new_objects[idx_db].catchup += 1
                        if self.new_objects[idx_db].catchup >= self.catchup:
                            new_objects[idx_obj].id = self.id_num
                            self.tracking_objects.append(new_objects[idx_obj])
                            self.id_num += 1

                for each in self.new_objects:
                    if each.catchup >= self.catchup:
                        self.new_objects.remove(each)
        """
        for obj in objects:
            if obj.id == -1:
                obj.id = self.id_num
                self.tracking_objects.append(obj)
                self.id_num += 1
        # Check for timeout items in the DB and delete them
        for i, db in enumerate(self.tracking_objects):
            #print(time.monotonic() - db.time)
            if time.monotonic() - db.time >= self.timeout_threshold:
                self.tracking_objects.pop(i)
        result = {'det': [], 'rec': []}
        for obj in self.tracking_objects:
            result['det'].append(obj.pos)
            result['rec'].append(obj.id)
        
        return result
            


    def run(self, frame, **kwargs):
        #res = self.run_async(frame, det_threshold= 0.6, draw=True, )
        """
        Args:
            image(numpy.ndarray): image
            **kwargs: 
                draw(bool, optional): draw result on image, Default is True
        """
        draw = kwargs.get('draw', True)
        res = self.run_async(frame, det_threshold=0.6, draw= draw, )
        # Create cosine distance matrix and match objects in the frame and the DB
        if len(res['det']) == 0:
            return None
        objects = []
        for vec, box in zip(res['rec'], res['det']):
            objects.append(tracking_object(box, vec))
        hangarian = Munkres()
        dist_matrix = [[distance.cosine(obj_db.feature, obj_cam.feature)
                        for obj_db in self.tracking_objects] for obj_cam in objects]
        combination = hangarian.compute(dist_matrix)# Solve matching problem
        for idx_obj, idx_db in combination:
            # This object has already been assigned an ID
            if objects[idx_obj].id != -1:
                continue
            dist = distance.cosine(
                objects[idx_obj].feature, self.tracking_objects[idx_db].feature)
            if dist < self.dist_threshold:
                self.tracking_objects[idx_db].time = time.monotonic()
                self.tracking_objects[idx_db].pos = objects[idx_obj].pos
                self.tracking_objects[idx_db].feature = objects[idx_obj].feature
                objects[idx_obj].id = self.tracking_objects[idx_db].id
        """
        # Remove noise
        del hangarian
        if self.new_objects == []:
            for obj in objects:
                if obj.id == -1:
                    self.new_objects.append(obj)
        else:
            new_objects = []
            for obj in objects:
                if obj.id == -1:
                    new_objects.append(obj)
            new_hangarian = Munkres()
            dist_matrix = [[distance.cosine(obj_db.feature, obj_cam.feature)
                            for obj_db in self.new_objects] for obj_cam in new_objects]
            if dist_matrix:
                combination = new_hangarian.compute(dist_matrix)  # Solve matching problem
                for idx_obj, idx_db in combination:
                    dist = distance.cosine(
                        new_objects[idx_obj].feature, self.new_objects[idx_db].feature)
                    if dist < self.dist_threshold:
                        self.new_objects[idx_db].catchup += 1
                        if self.new_objects[idx_db].catchup >= self.catchup:
                            new_objects[idx_obj].id = self.id_num
                            self.tracking_objects.append(new_objects[idx_obj])
                            self.id_num += 1

                for each in self.new_objects:
                    if each.catchup >= self.catchup:
                        self.new_objects.remove(each)
        """
        for obj in objects:
            if obj.id == -1:
                obj.id = self.id_num
                self.tracking_objects.append(obj)
                self.id_num += 1
        # Check for timeout items in the DB and delete them
        for i, db in enumerate(self.tracking_objects):
            #print(time.monotonic() - db.time)
            if time.monotonic() - db.time >= self.timeout_threshold:
                self.tracking_objects.pop(i)
                
        result = {'det': [], 'rec': []}
        for obj in self.tracking_objects:
            result['det'].append(obj.pos)
            result['rec'].append(obj.id)

        if draw:
            for obj in self.tracking_objects:
                id = obj.id
                xmin, ymin, xmax, ymax = obj.pos
                #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255, 255), 1)
                cv2.putText(frame, str(id), (xmin, ymin - 2), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)

        return result 
