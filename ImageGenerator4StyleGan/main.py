import glob
import logging

import cv2

import io
import os

from google.cloud import vision
from google.cloud import vision_v1

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials.json'
vision_client = vision.ImageAnnotatorClient()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
face_cascade3 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_file_paths = glob.glob("videos/*")

files = glob.glob("images/*.jpg")
latest_number = 0
for file in files:
    image_num = int(file.replace("images\\", "").replace(".jpg", ""))
    if image_num > latest_number:
        latest_number = image_num

logging.info(f"latest number is {latest_number}")

frame_count = latest_number
save_count = 0
for video_path in video_file_paths:
    print(f"tryin to open {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        exit(0)

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    while True:
        ret, frame = cap.read()
        if ret:
            if save_count % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                faces1 = face_cascade1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                faces3 = face_cascade3.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                if len(faces) != 0 or len(faces1) != 0 or len(faces2) != 0 or len(faces3) != 0 or len(eyes) != 0:
                    continue

                cv2.imwrite("temp.jpg", frame)
                with io.open("temp.jpg", 'rb') as image_file:
                    content = image_file.read()
                image = vision_v1.types.Image(content=content)
                response = vision_client.face_detection(image=image)
                if len(response.face_annotations) != 0:
                    print(f"found {len(response.face_annotations)} in {frame_count}: {video_path}")
                    continue

                path = f"images/{frame_count}.jpg"
                # frame = cv2.resize(frame, (512, 512))
                cv2.imwrite(path, frame)
                # print(f"save image, file name {path}")
                frame_count = frame_count + 1
        else:
            print(f"{video_path} process has finished!!")
            cap.release()
            break
        save_count = save_count + 1
