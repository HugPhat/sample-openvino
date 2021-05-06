import enum 
from types import MethodType

from libs import *
import config 

class TYPE(enum.Enum):
    FACE = 0
    PERSON = 1
    BOTH = -1


class app_run:
    def __init__(self, face, person, type, run) -> None:
        self.face_age_gender = face
        self.person_reid = person 
        self._type = type
        self.run = run
    
    def reset(self):
        if self.person_reid:
            self.person_reid.reset()

    @classmethod
    def face_age_gender(cls):
        _face = face_age_gender(det=config.FACE_DET, reg=config.FACE_REC, 
                            device=config.DEVICE, color=config.FACE_COLOR)
        _person = None 
        _type = TYPE.FACE.name
        run = _face.run
        return cls(_face, _person, _type, run)
    @classmethod
    def person_reid(cls):
        _person = person_reid(det=config.PERSON_DET, reg=config.PERSON_REID, 
                            device=config.DEVICE, color=config.PERSON_COLOR)
        _face = None 
        _type = TYPE.PERSON.name
        run = _person.run
        return cls(_face, _person, _type, run)
    @classmethod
    def both(cls):
        _person = person_reid(det=config.PERSON_DET, reg=config.PERSON_REID, 
                            device=config.DEVICE, color=config.PERSON_COLOR)
        _face = face_age_gender(det=config.FACE_DET, reg=config.FACE_REC, 
                            device=config.DEVICE, color=config.FACE_COLOR) 
        _type = TYPE.BOTH.name

        def run(image, draw):
            res = {'face' : None, 'reid': None}
            res['reid'] = _person.run(image, draw=draw)
            res['face'] = _face.run(image, draw=draw)
            return res 

        return cls(_face, _person, _type, run)