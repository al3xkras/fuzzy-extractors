import face_recognition.api as fr
import numpy as np
from PIL.Image import Image
from typing import SupportsIndex

import face_recognition
import cv2

Video = cv2.VideoCapture


class VideoFaceExtractor:
    def __init__(self, video: Video = None):
        self.video = video

    def load_file(self, path: str):
        self.video = Video(path)

    def iterate_images(self, image_consumer):
        video = self.video
        success, image = video.read()
        count = 0
        while success:
            image_consumer(image)
            success, image = video.read()
            count += 1

    @classmethod
    def save_image(cls, image: np.ndarray, file_name: str):
        cv2.imwrite(file_name, image)



class FaceVectorExtractor:
    def __init__(self, *args):
        pass

    @classmethod
    def get_face_bounding_box(cls, img: Image) -> tuple[int, int, int, int]:
        """
        Returns a bounding box of a human face in an image
        (if an image contains >1 or 0 faces, raise a runtime Exception)
        """
        img = cls.img_to_arr(img)

        boxes = fr.face_locations(img)
        if len(boxes) > 1:
            raise Exception("more than 1 face detected")
        if len(boxes) == 0:
            raise Exception("no face detected")
        return boxes[0]

    @staticmethod
    def img_to_arr(img: Image, mode="RGB"):
        return np.array(img.convert(mode))

    @classmethod
    def get_face_image(cls, img: Image) -> Image:
        """
        :return: cropped to a face bounding box image
        """
        bbox = cls.get_face_bounding_box(img)
        print(bbox)
        bbox = bbox[3], bbox[0], bbox[1], bbox[2]
        return img.crop(bbox)


class FuzzyExtractorFaceRecognition:
    """
    In order for fuzzy extractors to generate strong keys from biometric and other noisy data, cryptography paradigms will be applied to this biometric data. These paradigms:

    (1) Limit the number of assumptions about the content of the biometric data
     (this data comes from a variety of sources; so, in order to avoid exploitation
     by an adversary, it's best to assume the input is unpredictable).

    (2) Apply usual cryptographic techniques to the input. (Fuzzy extractors convert
     biometric data into secret, uniformly random, and reliably reproducible random strings.)

    These techniques can also have other broader applications for other type
    of noisy inputs such as approximative data from human memory, images used
    as passwords, and keys from quantum channels.[2] Fuzzy extractors also
    have applications in the proof of impossibility of the strong notions of
    privacy with regard to statistical databases.[3]
    """

    def __init__(self, conf_int=0.01, min_images=5, key_size=128):
        self.conf_int = conf_int
        self.min_images = min_images
        self.key_size = key_size

    def preprocess_images(self, images: SupportsIndex[np.ndarray]) -> list[np.ndarray]:
        pass

    def get_p_val(self, images: SupportsIndex[np.ndarray]) -> list[np.ndarray]:
        pass

    def hash_primary(self, images: SupportsIndex[np.ndarray]) -> bytes:
        """
        create a primary hash value, based on the p-value, and cropped images
        - The hash function should be collision resistant
          (hash value of similar within the confidence level faces should be very similar, but not equal)
         - Given a hash value, it should not be possible to retrieve the actual face landmarks
           (first image resistance)
         - The hash is not required to be 2-nd image resistant:
          (given face image, it is always possible
          to find a different face image with equal or similar hash)
        """
        pass

    def hash_secondary(self, hash_primary: bytes) -> bytes:
        """
         create a secondary hash value, based on the primary hash value: (possible algorithms: SHA256)
         - map similar within a confidence interval primary hash codes to equal secondary codes
         - collision resistance for hash codes, that are not similar within the confidence level
         - first preimage resistance
         - second preimage resistance
        """
        pass

    def hash_expansion(self, hash_secondary: bytes) -> bytes:
        """
        expand the hash value to the required key length
        """
        pass

    def generate_private_key(self, image_sequence: SupportsIndex[np.ndarray]) -> bytes:
        pass
