from unittest import TestCase

from PIL.Image import Image

from src.commons.testing import TestCases


class TestFaceVectorExtractor(TestCase):
    def test_get_face_bounding_box(self):
        img,tag = TestCases.randomImageTagged()
        print(tag)
        img.close()

    def test_get_face_image(self):
        self.fail()
