import os
import random

import PIL.Image
from PIL.Image import Image
import os

class TestCases:
    image_path = "../../images/"
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
        return PIL.Image.open(cls.image_path+img_file), tag

    @classmethod
    def clearCache(cls):
        cls._dir_cache = {}
