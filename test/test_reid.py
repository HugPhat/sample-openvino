import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libs.reid import person_reid


recog = r'models\person\reid\person-reidentification-retail-0288'
det = r'models\person\detection\person-detection-retail-0013'

model = person_reid(det=det, reg=recog, device='CPU', color=(255, 0, 50))
model.rect_thickness = 2
model.run_video(vid='videos/t4.mp4')
