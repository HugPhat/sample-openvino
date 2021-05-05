import os 
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libs.face import face_age_gender

recog = r'models\face\age_gender\age-gender-recognition-retail-0013'
det = r'models\face\detection\face-detection-0204'

model = face_age_gender(det= det, reg= recog, device='CPU', color= (255,0,255))

model.run_video(vid='videos/t1.mp4')