import math
import random

import PIL.Image
import face_recognition.api as fr
import numpy as np
from PIL.Image import Image
import cv2
import hashlib
from itertools import combinations

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
        if isinstance(img, np.ndarray):
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

    def __init__(self,
                 min_images=5,
                 d=0.03,
                 std_max=0.03,
                 alpha=0.5,
                 n_tests=500,
                 sample_size=0.7,
                 p_a_min=0.6,
                 max_unique_hashes=3):
        """
        :param min_images: min images to generate the hash (after preprocessing and rejecting outliers)
        :param d: the accuracy parameter (a positive float value, should be in range: (0,0.25])
        :param std_max: standard error maximum (if the std is > std_max, the input data will be rejected)
        :param alpha: the maximum percentage of face vector coordinates that can exceed the std_max threshold.
            (otherwise the input data will be rejected)
        :param n_tests: amount of independent tests to perform (hash values to create)
        :param sample_size: sample size of each test
        :param p_a_min: If a hash value is produced with a probability >= p_a_min,
            it will be returned by the method. Otherwise, the entropy of the face vectors will be taken into account.
        :param max_unique_hashes: max unique hash values allowed to accept the input data
        """
        self.min_images = min_images
        self.min_vectors = int(min_images * 0.8)
        self.d = d
        self.std_max = std_max
        self.alpha = alpha
        self.n_tests = n_tests
        self.sample_size = sample_size
        self.p_a_min = p_a_min
        self.max_unique_hashes = max_unique_hashes
        self.key_size_bytes=32

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

    @staticmethod
    def _get_outliers(data, std_max=0.5):
        """
        :param data: the input data (np.array or list) of tuples (items at index 0 are numeric)
        :param std_max: the maximum variance to determine outliers: values
            that exceed the maximum variance will be removed from the sample
        :return: a list of outliers that exceed the var_max parameter
        """
        if std_max <= 0:
            return data
        filter_ = np.array([x[0] for x in data])
        filter_ = abs(filter_ - np.mean(filter_)) >= std_max * np.std(filter_)
        rejected = [data[i] for i in range(len(data)) if filter_[i]]
        return rejected

    def reject_face_vector_outliers(self, face_vectors: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
        """
        Reject 'outlier' images (such images, that have a comparably high standard deviation, and can not be used for the
        generation of private key).
        1. Compute statistics for each pair of the input vectors (O(n^2))
        2. Find out, which pairs should be rejected (as the sample outliers)
        3. Compute the list of vectors to remove by their frequency in the list of rejected pairs
        4. Reject outliers from the list of vectors, formed in the previous step (by their number of occurrences).
            (in this case, outliers can only have higher values than the sample mean)
        5. If the percentage of outliers is too high, reject the input
            (unable to build the private key for the given set of images)
        5. Otherwise, return the list of vectors, that does not contain outliers.
        """
        sample = []

        def mean_diff(v1: np.ndarray, v2: np.ndarray) -> float:
            return (abs(v1 - v2)).mean()

        for i, j in combinations(range(len(face_vectors)), 2):
            a, b = face_vectors[i], face_vectors[j]
            f_ = mean_diff(a, b)
            # print("face vectors statistic:",i,j,f_)
            sample.append([f_, i, j])

        vec_sample = {}
        for _, i, j in self._get_outliers(sample):
            vec_sample[i] = vec_sample.get(i, 0) + 1

        vec_sample = self._get_outliers([(vec_sample[x], x) for x in vec_sample])
        vec_sample = set([x[1] for x in vec_sample])

        # print(len(sample_))
        return np.array([x for i, x in enumerate(face_vectors) if i not in vec_sample])

    def get_image_statistics(self, face_vectors: np.ndarray[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        :param face_vectors: a list of face vectors
        :return: tuple(list of image means for each coordinate, list of image std for each coordinate)
        """
        img_std = np.array([face_vectors[:, i].std() for i in range(face_vectors.shape[1])], dtype=float)
        img_mean = np.array([face_vectors[:, i].mean() for i in range(face_vectors.shape[1])], dtype=float)
        return img_mean, img_std

    def _hash_primary(self, face_vectors: np.ndarray[np.ndarray]) -> bytes:
        """
        Create a primary hash value, based on the list of face vectors
        - The hash function is not collision resistant
          (the hash value of 'similar' face vectors should be equal)
         - Given the hash value, it might be possible to retrieve the actual face landmarks
           (non-resistant to the first preimage)
         - The hash function is not 2-nd preimage resistant:
           (given face vector, it is always possible
            to find a different face vector with equal or similar hash)

        Implementation description:
        1. Consider a hypercube with a given edge length in the 128-dimensional space (D) with default
            Euclidean metrics
        2. D is uniformly divided into hypercubes, without gaps
        3. Then, if a certain face vector is located on a hypercube edge,
            the program should not compute the hash value and will raise an Exception instead.
        4. Otherwise, the hash value returned by this function
            is equal to the center of a hypercube, that contains the face vector
        5. If the standard deviation of the set of face vectors is high, the method should raise an Exception
            (1)(the hash value may not be created for the given images)
        6. A byte representation of the hypercube center is extended to a specified length (32 bytes by default)
        7. The byte representation is returned

        => As the result, this hash method maps similar face vectors into equal hash values with a high probability
            (1)(the actual score of this model is measured in the test cases)
            (2)(if the list of face vectors does not contain outliers)

        This method is not collision resistant, and is not a secure cryptographic hash function.
        It is a Monte-Carlo algorithm to estimate the actual private key of a person, therefore it has a fixed
            execution time, however it may output incorrect answers with a certain probability
            todo (1)(this probability is measured in the test cases, contained in this application)
        """
        img_mean, img_std = self.get_image_statistics(face_vectors)
        stat = sum(x > self.std_max for x in img_std)
        if stat > self.alpha * len(img_std):
            print(stat, img_std)
            raise ValueError("Std of the images provided is too high. Unable to build a safe primary hash: %d" % stat)

        def f(val: float):
            k = int(val / self.d)
            res = val - k * self.d
            if res == 0:
                raise ValueError
            return (k + 0.5) * self.d

        f = np.vectorize(f)
        actual_landmarks = f(img_mean).tobytes('C')
        return self.hash_format(actual_landmarks)

    def hash_format(self, key: bytes) -> bytes:
        """
        Expand the hash value to the required key length
        """
        if len(key) == self.key_size_bytes:
            return key
        elif len(key) > self.key_size_bytes:
            return key[:self.key_size_bytes]
        key = key + key[:(self.key_size_bytes - len(key))]
        assert len(key) == self.key_size_bytes
        return key

    @staticmethod
    def entropy(bytez: bytes) -> float:
        """
        :return: Shannon's entropy for the input byte sequence
        """
        bytes_dict = dict()
        for b in bytez:
            bytes_dict[b] = bytes_dict.get(b, 0) + 1
        for b in bytes_dict:
            bytes_dict[b] /= len(bytez)
        return -sum(x * math.log2(x) for x in bytes_dict.values())

    def hash_primary(self, face_vectors: np.ndarray[np.ndarray]) -> bytes:
        """
        Probability-based error correction paired with the _hash_primary method:
            This method is implemented in order to avoid possible errors, when face vectors are located close to a
            side of a specific hypercube. Because Euclidean metrics is naturally used by the face_recognition model
            author to compare face vectors, the _hash_primary method may sometimes
            map values to an incorrect hypercube center, which might cause mapping mistakes.
            This can be avoided by using probabilistic methods, implemented in this function:
        1. Perform X independent tests and write unique hashes to a collection.
        2. Select a random sample from the population of face vectors
        3. Select top-2 hashes that have the highest probability of occurrence
        4. If the amount of unique hash values is > $len_hashes_un_max$, reject the input
        4. If any value has a probability of occurrence p>=p_a_min, return the hash code
        5. Otherwise, find a hash value with the highest entropy and return it.
            (if both values have equal entropy, return a value which has a higher probability of occurrence)
        """

        def dict_t2(d: dict):
            out = [None, None]
            occurrences = [0, 0]
            for x in d:
                if d[x] > occurrences[0]:
                    occurrences[0] = d[x]
                    out[0] = x
                elif d[x] > occurrences[1]:
                    occurrences[1] = d[x]
                    out[1] = x
            return out, occurrences

        size = int(len(face_vectors) * self.sample_size)
        hashes_un = dict()
        face_vectors = list(face_vectors)
        for _ in range(self.n_tests):
            sample = np.array(random.sample(face_vectors, size))
            hash_val = self._hash_primary(sample)
            hashes_un[hash_val] = hashes_un.get(hash_val, 0) + 1

        if len(hashes_un) > self.max_unique_hashes:
            raise ValueError("input data rejected")

        hashes, vals = dict_t2(hashes_un)

        if vals[0] / sum(vals) >= self.p_a_min:
            return hashes[0]
        if vals[1] / sum(vals) >= self.p_a_min:
            return hashes[1]
        e1 = self.entropy(hashes[0])
        e2 = self.entropy(hashes[1])
        if e1 == e2:
            return hashes[0] if vals[0] > vals[1] else hashes[1]
        if e1 > e2:
            return hashes[0]
        return hashes[1]

    def hash_secondary(self, hash_primary: bytes) -> bytes:
        """
         Create a secondary hash value, based on the primary hash value: (possible algorithms: SHA256)
         - map similar within a confidence interval primary hash codes to equal secondary codes
         - collision resistance for hash codes, that are not similar within the confidence level
         - first preimage resistance
         - second preimage resistance

         This method uses built-in implementation of the SHA256 secure hash algorithm.
        """
        sha256 = hashlib.sha256()
        sha256.update(hash_primary)
        return sha256.digest()

    def generate_private_key(self, images: np.ndarray[np.ndarray]) -> bytes:
        vectors = self.preprocess_images(images)
        vectors = self.reject_face_vector_outliers(vectors)
        return self.hash_secondary(self.hash_primary(vectors))
