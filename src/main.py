
from src.commons.face_recognition import FaceVector

if __name__ == '__main__':
    import face_recognition

    image = face_recognition.load_image_file("your_file.jpg")
    face_landmarks_list = face_recognition.face_landmarks(image)