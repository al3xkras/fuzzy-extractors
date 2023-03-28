import face_recognition.api as fr
import numpy as np
from src.commons.face_recognition import FaceVector

from PIL.Image import Image


class FaceVectorExtractor:
    def __init__(self, *args):
        pass

    @staticmethod
    def get_face_bounding_box(img: Image) -> tuple[int, int, int, int]:
        """
        Returns a bounding box of a human face in an image
        (if an image contains >1 or 0 faces, raise a runtime Exception)
        """
        img = np.array(img.convert("RGB"))

        boxes = fr.face_locations(img)
        if len(boxes)>1:
            raise Exception("more than 1 face detected")
        if len(boxes)==0:
            raise Exception("no face detected")
        return boxes[0]


    @classmethod
    def get_face_image(cls, img: Image) -> Image:
        """
        :return: cropped to a face bounding box image
        """
        bbox = cls.get_face_bounding_box(img)
        return img.crop(bbox)

