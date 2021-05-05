import numpy as np 

from ..base.network import nn 
from ..base.base import base

class face_age_gender(base):

    def __init__(self, det:str, reg:str, device:str, color:tuple):
        super(face_age_gender, self).__init__(det, reg, device, color)
        
    def run(self, image):
        return 

    def recog(self, image):
        output = self.recognizer.async_infer(image)
        age= output['age_conv3'].flatten()
        genre = output['prob'].flatten()
        age_pred = int(age*100)
        genre_pred = np.argmax(genre)

        return age_pred, genre_pred

    def draw_recog(self, image, recog_res):
        print(recog_res)
        return 
        
    def infer_video(self, frame):
        self.run_async(frame)

    
