import numpy as np 
import cv2 
from ..base.network import nn 
from ..base.base import base

class face_age_gender(base):

    def __init__(self, det:str, reg:str, device:str, color:tuple):
        super(face_age_gender, self).__init__(det, reg, device, color)
        self.gender = ['Female', 'Male']
    def run(self, image):
        return 

    def recog(self, image):
        output = self.recognizer.async_infer(image)
        age = output['age_conv3'].buffer.flatten()
        genre = output['prob'].buffer.flatten()
        age_pred = int(age*100)
        genre_pred = np.argmax(genre)
        return age_pred, genre_pred

    def draw_recog(self, image, box, recog_res, **kwargs):
        age, gender = recog_res
        xmin, ymin, xmax, ymax = [int(each) for each in list(box)]
        text = f'{self.gender[gender]}age {str(age)}'
        cv2.putText(image, text, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.color, 1, cv2.LINE_AA)
        return 
        
    def infer_video(self, frame):
        self.run_async(frame, det_threshold=0.6,
                       draw=True, callback=self.draw_recog)

    
