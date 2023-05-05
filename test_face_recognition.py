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

fx = FuzzyExtractorFaceRecognition(min_images=20)
fun = lambda person, n=20: fx.preprocess_images(np.array(random.sample(TestCases.getImagesByTagPrefix(person), n)))

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
            elon = fx.preprocess_images(elon)
            c.cache_object(elon, cache_name)
        print(elon[0])

    def test_primary_hash_equality_for_the_same_person(self):
        c=Cache
        cache_names=[
            "test_primary_hash_elon1",
            "test_primary_hash_elon2"
        ]
        face_vectors = c.get_cached_object(cache_names[0])
        face_vectors1 = c.get_cached_object(cache_names[1])

        if face_vectors is None:
            face_vectors = fun("ElonMusk", 30)
            face_vectors1 = fun("ElonMusk", 30)
            c.cache_object(face_vectors,cache_names[0])
            c.cache_object(face_vectors1,cache_names[1])

        elon = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors))
        elon1 = fx.hash_primary(fx.reject_face_vector_outliers(face_vectors1))

        print(elon)
        print(elon1)

        self.assertEqual(elon, elon1)

    def test_primary_hash_inequality_for_different_people(self):

        elon = fx.hash_primary(fx.reject_face_vector_outliers(fun("ElonMusk", 20)))
        nile = fx.hash_primary(fx.reject_face_vector_outliers(fun("NileRed", 20)))

        print(elon)
        print(nile)

        self.assertNotEqual(elon, nile)

    def test__hash_primary(self):
        name = "ElonMusk"
        trials = 1500
        population_size = 55
        face_vectors = [x for x in fun(name, population_size)]
        samp_size = int(len(face_vectors) * 0.7)
        unique_hashes = dict()
        for i in range(trials):
            samp = random.sample(face_vectors, samp_size)
            elon = fx._hash_primary(fx.reject_face_vector_outliers(samp))
            unique_hashes[elon] = unique_hashes.get(elon, 0) + 1
        print("Unique hashes (for %s):\n" % name, unique_hashes, "\n")

    def test_hash_primary(self):
        name = "ElonMusk"
        population_size = 55
        face_vectors = fun(name, population_size)
        face_vectors = fx.reject_face_vector_outliers(face_vectors)
        hash_val = fx.hash_primary(face_vectors)
        print(hash_val)

    def test_remove_face_vector_outliers(self):

        elon = fun("NileRed", 20)
        print("length before rejecting outliers: ", len(elon))

        elon1 = fx.reject_face_vector_outliers(elon)
        print("length after rejecting outliers: ", len(elon1))

        elon2 = fx.reject_face_vector_outliers(elon1)
        print("length after rejecting twice: ", len(elon2))

    def test_generate_private_key(self):
        pass
