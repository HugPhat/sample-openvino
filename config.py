FACE_DET_THRESH = 0.5 

DEVICE = 'CPU'

PERSON_REID = 'models/person/reid/person-reidentification-retail-0288'
PERSON_DET = 'models/person/detection/person-detection-retail-0013'

FACE_REC = 'models/face/age_gender/age-gender-recognition-retail-0013'
FACE_DET = 'models/face/detection/face-detection-0204'

FACE_COLOR = (255, 12, 3)
PERSON_COLOR = (0, 124, 123)

#######################################


class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    JSON_SORT_KEYS = False
    #SECRET_KEY = 'this-really-needs-to-be-changed'


class ProductionConfig(Config):
    DEBUG = False
    PORT = 80
    HOST = '0.0.0.0'

class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    PORT= 5000
    HOST = 'localhost'


class TestingConfig(Config):
    TESTING = True