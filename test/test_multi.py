import os 
import sys 
import cv2 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import app_run

model = app_run.both()

vid = 'videos/t4.mp4'
cap = cv2.VideoCapture(vid)
while True:
    has_frame, frame = cap.read()
    if not has_frame:
        cv2.destroyAllWindows()
        break
    res = model.run(frame, draw= True)
    cv2.namedWindow('test multi', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('test multi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break