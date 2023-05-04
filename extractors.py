import sys

import PIL.Image
import face_recognition.api as fr
import numpy as np
from PIL.Image import Image
import cv2
import hashlib

Video = cv2.VideoCapture


class FrameIterator:
    def __init__(self, video: Video | str = None):
        if isinstance(video, Video):
            self.video = video
        else:
            self.video = Video(video)

    def load_file(self, path: str):
        self.video = Video(path)

    def iterate_images(self, image_consumer):
        video = self.video
        success, image = video.read()
        count = 0
        while success:
            success, image = video.read()
            success = success and image_consumer(image)
            count += 1

    @classmethod
    def save_image(cls, image: np.ndarray, file_name: str):
        cv2.imwrite(file_name, image)


class FaceVectorExtractor:
    def __init__(self, *args):
        pass

    @classmethod
    def get_face_bounding_box(cls, img: Image | np.ndarray) -> tuple[int, int, int, int]:
        """
        Returns a bounding box of a human face in an image
        (if an image contains >1 or 0 faces, raise a runtime Exception)
        """
        if isinstance(img, Image):
            img = cls.img_to_arr(img)

        boxes = fr.face_locations(img)
        if len(boxes) > 1:
            raise ValueError("more than 1 face detected")
        if len(boxes) == 0:
            raise ValueError("no face detected")
        return boxes[0]

    @staticmethod
    def img_to_arr(img: Image, mode="RGB"):
        if isinstance(img,np.ndarray):
            return img
        return np.array(img.convert(mode))

    @classmethod
    def get_face_image(cls, img: Image | np.ndarray) -> Image | np.ndarray:
        """
        :return: cropped to a face bounding box image
        """
        bbox = cls.get_face_bounding_box(img)
        print(bbox)
        if isinstance(img, Image):
            bbox = bbox[3], bbox[0], bbox[1], bbox[2]
            return img.crop(bbox)
        elif isinstance(img, np.ndarray):
            return img[bbox[0]:bbox[2], bbox[3]:bbox[1]]


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

    def __init__(self, conf_int=0.01, min_images=5, key_size_bytes=32):
        self.conf_int = conf_int
        self.min_images = min_images
        self.key_size_bytes = key_size_bytes
        self.d = 0.02
        self.std_thr = 0.02
        self.mean_thr = 0.04
        self.alpha = 0.5

    def preprocess_images(self, images: np.ndarray[np.ndarray | Image]) -> np.ndarray[np.ndarray]:
        """
        1. convert images to numpy (mode: RGB)
        2. locate faces; remove images that do not contain any face
        3. remove images that contain multiple faces
        4. crop image to a face rectangle
        5. return processed list of faces
        """
        if len(images) == 0:
            return images

        g = lambda x: fr.face_encodings(x)[0]

        def f(item):
            if isinstance(item, Image):
                item = FaceVectorExtractor.img_to_arr(item)
            try:
                FaceVectorExtractor.get_face_bounding_box(item)
                item = g(item)
            except ValueError | IndexError:
                return None
            return item
        lst = [f(x) for x in images]
        lst = [x for x in lst if x is not None]
        return np.array(lst)

    def get_image_statistics(self, face_vectors: np.ndarray[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        input: a set of face vectors
        output: array of means, array of standard deviations
        """
        img_std = np.array([face_vectors[:, i].std() for i in range(face_vectors.shape[1])], dtype=float)
        img_mean = np.array([face_vectors[:, i].mean() for i in range(face_vectors.shape[1])], dtype=float)
        return img_mean, img_std

    def hash_primary(self, images_processed: np.ndarray[np.ndarray]) -> bytes:
        """
        create a primary hash value, based on the p-value, and cropped images
        - The hash function should be collision resistant
          (hash value of similar within the confidence level faces should be very similar, but not equal)
         - Given a hash value, it should not be possible to retrieve the actual face landmarks
           (first image resistance)
         - The hash is not required to be 2-nd image resistant:
          (given face image, it is always possible
          to find a different face image with equal or similar hash)

        used method:
        1. consider a hypercube of given radius r_0 \\gt 0 in 128-dimensional space (D) with default
        Euclidean metrics
        2. D is divided into packed spheres of given radius (with gaps)
        3. If a certain face vector
        is inside a gap, the program should not compute its hash and raise an Exception
        4. Otherwise, the hash value
        will be equal to the center of a sphere, where the face vector is located
        5. If the standard deviation of the
        image set is high at a fixed level, the method will raise an Exception (hash value can not be created for
        the given set)
        6. Byte representation of the sphere center is returned (128-dimensional vector) => primary
        hash will map similar faces with similarity coefficient determined by the sphere radius into equal hash
        values. It is obviously not collision resistant, and is not a secure cryptographic hash function.
        """
        img_mean, img_std = self.get_image_statistics(images_processed)
        stat=sum(x > self.std_thr for x in img_std)
        if stat > self.alpha * len(img_std):
            print(stat,img_std)
            raise ValueError("Std of the images provided is too high. Unable to build a safe primary hash: %d"%stat)

        def f(val: float):
            k = int(val / self.d)
            res = val - k * self.d
            if res == 0:
                raise ValueError
            return (k + 0.5) * self.d

        f = np.vectorize(f)

        actual_landmarks = f(img_mean)
        return self.hash_format(actual_landmarks.tobytes('C'))

    def hash_format(self, key: bytes) -> bytes:
        """
        expand the hash value to the required key length
        """
        # todo use the kupyna hash algo
        if len(key) == self.key_size_bytes:
            return key
        elif len(key) > self.key_size_bytes:
            return key[:self.key_size_bytes]
        key = key + key[:(self.key_size_bytes - len(key))]
        assert len(key) == self.key_size_bytes
        return key

    def hash_secondary(self, hash_primary: bytes) -> bytes:
        """
         Create a secondary hash value, based on the primary hash value: (possible algorithms: SHA256)
         - map similar within a confidence interval primary hash codes to equal secondary codes
         - collision resistance for hash codes, that are not similar within the confidence level
         - first preimage resistance
         - second preimage resistance

         This method uses built-in implementation of the SHA256 algorithm.
        """
        sha256 = hashlib.sha256()
        sha256.update(hash_primary)
        return self.hash_format(sha256.digest())

    def generate_private_key(self, images: np.ndarray[np.ndarray]) -> bytes:
        images = self.preprocess_images(images)
        return self.hash_secondary(self.hash_primary(images))
