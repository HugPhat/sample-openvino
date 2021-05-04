
class base:
    def __init__(self, **kwargs) -> None:
        pass

    def run(self, image):
        raise NotImplemented()

    def async(self, image):
        raise NotImplemented()
