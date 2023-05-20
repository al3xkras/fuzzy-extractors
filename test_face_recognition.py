import os.path
import random
from unittest import TestCase
import PIL.Image
import numpy as np
import face_recognition as fr

from testing import TestCases
from extractors import FaceVectorExtractor
from extractors import FuzzyExtractorFaceRecognition
from logger import Logger
from cache import Cache
from PIL import Image
from reedsolo import ReedSolomonError
fx = FuzzyExtractorFaceRecognition(min_images=20)

fun = lambda person, n=20: fx._preprocess_images(np.array(random.sample(TestCases.getImagesByTagPrefix(person), n)))


def fun1(person, n=20):
    images = np.array(random.sample(TestCases.getImagesByTagPrefix(person), n))
    return images, fx._preprocess_images(images)


print = lambda *o: Logger.info(" ".join(map(str, o))) if len(o) > 0 else Logger.info("\n")


class TestFaceVectorExtractor(TestCase):
    def test_img_opens(self):
        img, tag = TestCases.randomImageTagged()
        print(tag)
        img.close()

    def test_get_face_bounding_box(self):
        img, tag = TestCases.randomImageTagged()

        bbox = FaceVectorExtractor.get_face_bounding_box(img)
        self.assertIsNotNone(bbox)
        self.assertIsInstance(bbox, tuple)
        print(bbox)

        img.close()

    def test_get_face_image(self):
        img, tag = TestCases.randomImageTagged()
        face = FaceVectorExtractor.get_face_image(img)
        face.save("../../tmp/test_get_face_image1.png", "png")
        img.close()

    def test_get_face_vector(self):
        img, tag = TestCases.randomImageTagged()
        face = FaceVectorExtractor.get_face_image(img)
        face_arr = FaceVectorExtractor.img_to_arr(face)
        encodings = fr.face_encodings(face_arr)[0]
        print(encodings)
        print(len(encodings))

    def test_face_vector_similarity(self):
        img1, img2 = TestCases.getImagesByTagPrefix("NileRed")[:2]
        img3, img4 = TestCases.getImagesByTagPrefix("ElonMusk")[:2]

        i = 1

        def get_encodings(img):
            nonlocal i
            face = FaceVectorExtractor.get_face_image(img)
            PIL.Image.fromarray(face).save("/fuzzy/tmp/face%s.png" % i)
            i += 1
            arr = FaceVectorExtractor.img_to_arr(img)
            try:
                return fr.face_encodings(arr)[0]
            except:
                return None

        enc1 = get_encodings(img1)
        enc2 = get_encodings(img2)
        enc3 = get_encodings(img3)
        enc4 = get_encodings(img4)
        print(enc1, enc2, enc3, enc4)

        diff = abs(enc1 - enc2)
        diff1 = abs(enc3 - enc4)
        diff2 = abs(enc1 - enc3)

        print("(images contain the same person)")

        print("mean: ", diff.mean())
        print("std: ", diff.std())
        print("mean: ", diff1.mean())
        print("std: ", diff1.std())

        print()
        print("(images contain different people)")
        print("mean: ", diff2.mean())
        print("std: ", diff2.std())


class TestFuzzyExtractorFaceRecognition(TestCase):

    def test_init(self) -> FuzzyExtractorFaceRecognition:
        return FuzzyExtractorFaceRecognition(min_images=20)

    def test_preprocess_images(self):
        cache_name = "test_preprocess_images"
        c = Cache
        elon = c.get_cached_object(cache_name)
        if elon is None:
            elon = np.array(TestCases.getImagesByTagPrefix("ElonMusk")[:15])
            elon = fx._preprocess_images(elon)
            c.cache_object(elon, cache_name)
        print(elon[0])

    def test_primary_hash_similarity_for_the_same_person(self):
        fx = FuzzyExtractorFaceRecognition(max_unique_hashes=1000)
        c = Cache
        cache_names = [
            "test_primary_hash_elon1",
            "test_primary_hash_elon2"
        ]
        face_vectors = c.get_cached_object(cache_names[0])
        face_vectors1 = c.get_cached_object(cache_names[1])

        if face_vectors is None or face_vectors1 is None:
            face_vectors = fun("ElonMusk", 30)
            face_vectors1 = fun("ElonMusk", 30)
            c.cache_object(face_vectors, cache_names[0])
            c.cache_object(face_vectors1, cache_names[1])

        elon = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors))
        elon1 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors1))

        # equality is not required, because the input data does not match requirements
        print(elon)
        print(elon1)

    def test_primary_hash_similarity_for_the_same_person2(self):
        fx = FuzzyExtractorFaceRecognition(max_unique_hashes=1000)
        """
        - Given 3 different videos that contain the same person, check whether
            the primary hash is equal for all 3 videos
        """
        c = Cache
        cache_names = [
            "test_primary_hash_equality_for_the_same_person21",
            "test_primary_hash_equality_for_the_same_person22",
            "test_primary_hash_equality_for_the_same_person23"
        ]

        face_vectors = c.get_cached_object(cache_names[0])
        face_vectors1 = c.get_cached_object(cache_names[1])
        face_vectors2 = c.get_cached_object(cache_names[2])

        if face_vectors is None:
            face_vectors = fun("test1_1", 40)
            c.cache_object(face_vectors, cache_names[0])

        if face_vectors1 is None:
            face_vectors1 = fun("test1_2", 40)
            c.cache_object(face_vectors1, cache_names[1])

        if face_vectors2 is None:
            face_vectors2 = fun("test1_3", 40)
            c.cache_object(face_vectors2, cache_names[2])

        h1 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors))
        h2 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors1))
        h3 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors2))

        print(h1.hex())
        print(h2.hex())
        print(h3.hex())

    def test_primary_hash_inequality_for_different_people(self):
        fx = FuzzyExtractorFaceRecognition(max_unique_hashes=1000)
        c = Cache
        cache_names = [
            "test_primary_hash_elon3",
            "test_primary_hash_nile1"
        ]

        face_vectors_elon = c.get_cached_object(cache_names[0])
        face_vectors_nile = c.get_cached_object(cache_names[1])

        if face_vectors_elon is None or face_vectors_nile is None:
            face_vectors_elon = fun("ElonMusk", 30)
            face_vectors_nile = fun("NileRed", 30)
            c.cache_object(face_vectors_elon, cache_names[0])
            c.cache_object(face_vectors_nile, cache_names[1])

        elon = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors_elon))
        nile = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors_nile))

        print(elon.hex())
        print(nile.hex())

        self.assertNotEqual(elon, nile)

    def test__hash_primary(self):
        fx = FuzzyExtractorFaceRecognition()
        name = "ElonMusk"
        c = Cache
        cache_names = [
            "test_primary_hash_elon4"
        ]
        trials = 1500
        population_size = 50
        face_vectors = c.get_cached_object(cache_names[0])
        if face_vectors is None:
            face_vectors = fun(name, population_size)
            c.cache_object(face_vectors, cache_names[0])
        face_vectors = [x for x in face_vectors]
        samp_size = int(len(face_vectors) * 0.7)
        unique_hashes = dict()
        for i in range(trials):
            samp = random.sample(face_vectors, samp_size)
            elon = fx._hash_primary(fx.reject_face_vector_outliers(samp)).hex()
            unique_hashes[elon] = unique_hashes.get(elon, 0) + 1
        print("Unique hashes (for %s):\n" % name, unique_hashes, "\n")

    def test_remove_face_vector_outliers(self):
        fx = FuzzyExtractorFaceRecognition()
        c = Cache
        cache_names = [
            "test_remove_face_vector_outliers"
        ]
        cache = c.get_cached_object(cache_names[0])

        if cache is None:
            images, face_vectors = fun1("ElonMusk", 30)
            c.cache_object((images, face_vectors), cache_names[0])
        else:
            images, face_vectors = cache

        print("length before rejecting outliers: ", len(face_vectors))

        elon1 = fx.reject_face_vector_outliers(face_vectors)
        accepted1 = [
            images[i] for i, v in enumerate(face_vectors) if v in elon1
        ]
        print("length after rejecting outliers: ", len(elon1))

        elon2 = fx.reject_face_vector_outliers(elon1)
        accepted2 = [
            images[i] for i, v in enumerate(face_vectors) if v in elon2
        ]
        print("length after rejecting twice: ", len(elon2))

        def write_tmp_images(imgs, path: str):
            path = os.path.dirname(__file__) + path
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
            for i, img in enumerate(imgs):
                img_ = Image.fromarray(img).convert("RGB")
                img_.save(os.path.join(path, "image%d.png" % i))

        write_tmp_images(images, "/tmp/images/initial")
        write_tmp_images(accepted1, "/tmp/images/no_outliers")
        write_tmp_images(accepted2, "/tmp/images/iterated_twice")

    def test_generate_check_symbols(self):
        fx = FuzzyExtractorFaceRecognition(max_unique_hashes=1000)
        """
        - Given 3 different videos that contain the same person, check whether
            the primary hash is equal for all 3 videos
        """
        c = Cache
        cache_names = [
            "test_primary_hash_equality_for_the_same_person21",
            "test_primary_hash_equality_for_the_same_person22",
            "test_primary_hash_equality_for_the_same_person23"
        ]

        face_vectors = c.get_cached_object(cache_names[0])
        face_vectors1 = c.get_cached_object(cache_names[1])
        face_vectors2 = c.get_cached_object(cache_names[2])

        if face_vectors is None:
            face_vectors = fun("test1_1", 40)
            c.cache_object(face_vectors, cache_names[0])

        if face_vectors1 is None:
            face_vectors1 = fun("test1_2", 40)
            c.cache_object(face_vectors1, cache_names[1])

        if face_vectors2 is None:
            face_vectors2 = fun("test1_3", 40)
            c.cache_object(face_vectors2, cache_names[2])

        h1 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors))
        check_symbols = fx.get_check_symbols(h1)

        h2 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors1))
        h3 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors2))

        print()
        print(h1.hex())
        print(h2.hex())
        print(h3.hex())
        print()
        print(check_symbols)

        print(len(check_symbols))

        h11 = fx.hash_secondary(h1,check_symbols)
        h12 = fx.hash_secondary(h2,check_symbols)
        h13 = fx.hash_secondary(h3,check_symbols)

        print(h11.hex())
        print(h12.hex())
        print(h13.hex())

    def test_generate_check_symbols1(self):
        fx = FuzzyExtractorFaceRecognition(max_unique_hashes=1000)

        c = Cache
        cache_names = [
            "test_primary_hash_elon3",
            "test_primary_hash_nile1"
        ]

        face_vectors_elon = c.get_cached_object(cache_names[0])
        face_vectors_nile = c.get_cached_object(cache_names[1])

        if face_vectors_elon is None or face_vectors_nile is None:
            face_vectors_elon = fun("ElonMusk", 30)
            face_vectors_nile = fun("NileRed", 30)
            c.cache_object(face_vectors_elon, cache_names[0])
            c.cache_object(face_vectors_nile, cache_names[1])

        h1 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors_elon))
        check_symbols1 = fx.get_check_symbols(h1)

        h2 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors_nile))
        check_symbols2 = fx.get_check_symbols(h2)

        print()
        print(h1.hex())
        print(h2.hex())
        print()
        print(check_symbols1)
        print()
        print(check_symbols2)

        h11 = b""
        h12=b""
        try:
            h11 = fx.hash_secondary(h1, check_symbols2)
            self.fail()
        except ReedSolomonError:
            pass
        try:
            h12 = fx.hash_secondary(h2, check_symbols1)
            self.fail()
        except ReedSolomonError:
            pass
        print("hash1:",h11.hex())
        print("hash2:",h12.hex())

    def test_hash_primary_probability_of_mistake(self):

        init_parameters = {}  # default parameters

    def test_generate_private_key(self):
        pass
