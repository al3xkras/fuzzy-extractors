import numpy as np

from extractors import FrameIterator, FaceVectorExtractor

names = [
    "NileRed",
    "ElonMusk",
    "test1_1",
    "test1_2",
    "test1_3",
]


def extract_video_faces(name: str):
    fi = FrameIterator("./videos/%s.mp4" % name)
    ex = FaceVectorExtractor()
    k = 5
    i = 0
    i_max = 50*k
    j = 0

    def consumer(img: np.ndarray):
        nonlocal i, j
        if i > i_max:
            return False
        i += 1
        if i % k != 0:
            return True
        try:
            ex.get_face_image(img)
            fi.save_image(img, "./images/%s%d.png" % (name, j))
            j += 1
        except ValueError:
            return True
        return True

    fi.iterate_images(consumer)


if __name__ == '__main__':
    extract_video_faces(names[3])