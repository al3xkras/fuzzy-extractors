import face_recognition as fr

from src.commons.face_recognition import FaceVector
from PIL.Image import Image
from typing import SupportsIndex



class FaceVectorExtractor:
    def __init__(self, *args):
        pass

    @staticmethod
    def get_face_bounding_box(img: Image) -> SupportsIndex[int, int, int, int]:
        """
        Returns a bounding box of a human face in an image
        (if an image contains >1 or 0 faces, raise a runtime Exception)
        """
        return fr.face_locations(img)

    @classmethod
    def get_face_image(cls,img: Image) -> Image:
        """
        :return: cropped to a face bounding box image
        """
        bbox = cls.get_face_bounding_box(img)
        return img.crop(bbox)

