from ..base.network import nn 
from ..base.base import base

class face_age_gender:

    def __init__(self, det:str, reg:str, device:str) -> None:
        self.detector = nn(det, device)

        self.recog = nn(reg, device)

    