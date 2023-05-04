from unittest import TestCase

from testing import TestCases
from extractors import FaceVectorExtractor

import face_recognition as fr


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
        img1 = TestCases.getImageByTag("NileRed100")
        img2 = TestCases.getImageByTag("NileRed500")

        img3 = TestCases.getImageByTag("stark1")
        img4 = TestCases.getImageByTag("stark1")

        i=1
        def get_encodings(img):
            nonlocal i
            face = FaceVectorExtractor.get_face_image(img)
            face.save("/fuzzy/tmp/face%d.png"%i)
            i+=1
            arr = FaceVectorExtractor.img_to_arr(face)

            return fr.face_encodings(arr)[0]

        enc1 = get_encodings(img1)
        enc2 = get_encodings(img2)
        enc3 = get_encodings(img3)
        enc4 = get_encodings(img4)
        # print(enc1)
        # print(enc2)

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
