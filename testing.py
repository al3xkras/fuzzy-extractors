import os
import random

import PIL.Image
import numpy as np
from PIL.Image import Image
from extractors import FaceVectorExtractor
import os


class TestCases:
    image_path = os.path.dirname(__file__) + "/images/"
    tmp_path = "tmp/"
    _dir_cache = {}

    @classmethod
    def listTestImages(cls) -> list[str]:
        return os.listdir(cls.image_path)

    @classmethod
    def randomImageTagged(cls) -> tuple[Image, str]:
        files = cls._dir_cache.get(cls.image_path, cls.listTestImages())
        if cls.image_path not in cls._dir_cache:
            cls._dir_cache[cls.image_path] = files
        img_file = files[random.randint(0, len(files) - 1)]
        tag = img_file.split(".")[0]
        return PIL.Image.open(cls.image_path + img_file), tag

    @classmethod
    def getImageByTag(cls, tag: str):
        return PIL.Image.open(cls.image_path + tag + ".png")

    @classmethod
    def getImagesByTagPrefix(cls, prefix: str) -> list[np.ndarray]:
        files = cls._dir_cache.get(cls.image_path, cls.listTestImages())
        img_lst = list()
        for img_file in files:
            if cls.image_path not in cls._dir_cache:
                cls._dir_cache[cls.image_path] = files
            tag = img_file.split(".")[0]
            if tag.startswith(prefix):
                img = PIL.Image.open(cls.image_path + img_file)
                img_lst.append(FaceVectorExtractor.img_to_arr(img))
                img.close()

        return img_lst

    @classmethod
    def clearCache(cls):
        cls._dir_cache = {}
