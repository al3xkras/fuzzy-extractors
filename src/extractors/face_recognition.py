import face_recognition

from .face_recognition import FaceVector

class FaceVectorExtractor:
    def __init__(self, bitmap):
        self.bitmap=bitmap

    def get_face_vector(self) -> FaceVector:
        pass

