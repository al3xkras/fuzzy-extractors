import numpy as np
import typer
from base64 import b64encode, b64decode

#app = typer.Typer()

from extractors import *

names = [
    "NileRed",
    "ElonMusk",
    "test1_1",
    "test1_2",
    "test1_3",
]


#@app.command()
def extract_video_faces(name: str, d_frames=5, max_images=50, prefix=False):
    if bool(prefix):
        name = "./videos/%s.mp4" % name
        print(name)
    fi = FrameIterator(name)
    ex = FaceVectorExtractor()
    k = int(d_frames)
    i = 0
    i_max = int(max_images) * k
    j = 0

    def consumer(img: np.ndarray):
        nonlocal i, j

        if i > i_max:
            return False
        i += 1
        if i % k != 0:
            return True
        print("Extracting image: %d" % j)
        try:
            ex.get_face_image(img)
            fi.save_image(img, "./images/%s%d.png" % (name, j))
            j += 1
        except ValueError:
            return True
        return True

    fi.iterate_images(consumer)


@app.command()
def fuzzy_extractor(
        video_path: str,
        check_symbols: str = None,
        salt: str = None,
        std_max=0.7,
        d=0.06,
        max_unique_hashes=-1,
        p_a_min=0.6,
        check_symbols_count=32,
        n_tests=250,
        sample_size=0.7,
        min_images=1,
        alpha=0.5):

    args = {
        "std_max":float(std_max),
        "d":float(d),
        "max_unique_hashes":int(max_unique_hashes),
        "p_a_min":float(p_a_min),
        "check_symbols_count":int(check_symbols_count),
        "n_tests":int(n_tests),
        "sample_size":float(sample_size),
        "min_images":int(min_images),
        "alpha":float(alpha)
    }

    fi = FrameIterator(video=video_path)
    extractor = FuzzyExtractorFaceRecognition(**args)
    j = 0
    images = []
    max_images = 5
    is_none = check_symbols is None
    salt = b64decode(salt) if salt is not None else None
    check_symbols = b64decode(check_symbols) if not is_none else None

    def consumer(img: np.ndarray):
        nonlocal j, images
        print("Extracting image: %d" % j)
        if j >= max_images:
            return False
        try:
            images.append(img)
            j += 1
        except ValueError:
            print("Interrupting image extractor")
            return True
        return True

    fi.iterate_images(consumer)
    images = np.array(images)

    print("Generating a private key:")
    out = extractor.recover_private_key(images, check_symbols, salt, log=True)
    if not is_none:
        print("[Rec] Recovered private key (Base64-encoded):")
        out = b64encode(out)
        print(out)
        return out
    print("[Gen] Private key and check symbols (Base64-encoded):")
    out = b64encode(out[0]), b64encode(out[1])
    print("Key: %s"%out[0])
    print("Check: %s"%out[1])
    return out


if __name__ == '__main__':
    #app()
    extract_video_faces("test1_4_a",prefix=True,d_frames=2)