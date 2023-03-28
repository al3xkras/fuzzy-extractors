from unittest import TestCase

from PIL.Image import Image

from src.commons.testing import TestCases
from src.extractors.extractors import FaceVectorExtractor

class TestFaceVectorExtractor(TestCase):
    def test_img_opens(self):
        img,tag = TestCases.randomImageTagged()
        print(tag)
        img.close()

    def test_get_face_bounding_box(self):
        img,tag = TestCases.randomImageTagged()

        bbox = FaceVectorExtractor.get_face_bounding_box(img)
        self.assertIsNotNone(bbox)
        self.assertIsInstance(bbox,tuple)
        print(bbox)

        img.close()

    def test_get_face_image(self):
        img,tag = TestCases.randomImageTagged()

        face = FaceVectorExtractor.get_face_image(img)
        face.save("../../tmp/test_get_face_image1.png","png")
        expected_after_crop=(0,0,face.width,face.height)
        print(expected_after_crop)
        img.close()
