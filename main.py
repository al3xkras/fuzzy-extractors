import numpy as np
import typer

app = typer.Typer()

from extractors import FrameIterator, FaceVectorExtractor

names = [
    "NileRed",
    "ElonMusk",
    "test1_1",
    "test1_2",
    "test1_3",
]


@app.command()
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
        print("Extracting image: %d"%j)
        try:
            ex.get_face_image(img)
            fi.save_image(img, "./images/%s%d.png" % (name, j))
            j += 1
        except ValueError:
            return True
        return True

    fi.iterate_images(consumer)


if __name__ == '__main__':
    app()
