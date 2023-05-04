import random
from unittest import TestCase

import PIL.Image
import numpy as np

from testing import TestCases
from extractors import FaceVectorExtractor

import face_recognition as fr

from extractors import FuzzyExtractorFaceRecognition

fx = FuzzyExtractorFaceRecognition(min_images=20)
fun = lambda person,n=20: fx.preprocess_images(np.array(random.sample(TestCases.getImagesByTagPrefix(person), n)))


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
        elon = np.array(TestCases.getImagesByTagPrefix("NileRed")[:15])
        elon = fx.preprocess_images(elon)
        print(elon[0])

    def test_primary_hash_equality_for_the_same_person(self):

        elon = fx.hash_primary(fun("ElonMusk"))
        elon1 = fx.hash_primary(fun("ElonMusk"))

        print(elon)
        print(elon1)

        self.assertEqual(elon, elon1)

    def test_primary_hash_inequality_for_different_people(self):

        elon = fx.hash_primary(fun("ElonMusk"))
        nile = fx.hash_primary(fun("NileRed"))

        print(elon)
        print(nile)

        self.assertNotEqual(elon, nile)

    def test_hash_primary(self) -> bytes:

        elon = np.array(random.sample(TestCases.getImagesByTagPrefix("ElonMusk"), 20))

        elon1 = np.array(random.sample(TestCases.getImagesByTagPrefix("ElonMusk"), 20))

        nile = np.array(random.sample(TestCases.getImagesByTagPrefix("NileRed"), 20))

        elon = fx.preprocess_images(elon)
        elon1 = fx.preprocess_images(elon1)
        nile = fx.preprocess_images(nile)

        hash_elon = fx.hash_primary(elon)
        hash_elon1 = fx.hash_primary(elon1)
        hash_nile = fx.hash_primary(nile)

        self.assertNotEqual(hash_elon, hash_nile)

        print(hash_elon)
        print(hash_elon1)
        print(hash_nile)

    def test_distil_face_vector_outliers(self):

        elon = fun("NileRed",20)
        print("length before rejecting outliers: ",len(elon))

        elon1=fx.distil_face_vector_outliers(elon)
        print("length after rejecting outliers: ", len(elon1))

        elon2 = fx.distil_face_vector_outliers(elon1)
        print("length after rejecting twice: ",len(elon2))



