from libs.face import face_age_gender
from libs.reid import person_reid

import config 


face = face_age_gender(det=config.FACE_DET, reg=config.FACE_REC, 
                            device='CPU', color=config.FACE_COLOR)
person = person_reid(det=config.PERSON_DET, reg=config.PERSON_REID, 
                            device='CPU', color= config.PERSON_COLOR)

