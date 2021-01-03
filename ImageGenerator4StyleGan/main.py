import glob
import io
import logging
import os
import shutil

import cv2
import numpy as np
from google.cloud import vision
from google.cloud import vision_v1

# path to credintial for google api
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\mail\Desktop\stylegan2-ada\ImageGenerator4StyleGan\credentials.json'

IMAGE_ROOT_DIR = "user/preprocess_images"
IMAGE_RESULT_DIR = "user/images"
SAVE_INTERVAL = 3


def make_directory(name="", permission=True):
    # ask making directory or continue previous run
    try:
        os.mkdir(name)
    except FileExistsError:
        if permission:
            logging.warning(f"Already exists {name} directory... remove direcotry? [y/n]")
            answer = input()
            if answer == "y":
                logging.warning(f"remove {name} direcotry make directry {name}")
                shutil.rmtree(name, ignore_errors=True)
                os.mkdir(name)
        else:
            logging.warning(f"ignore make_directory function, since the directory already exists")


def get_latest_number(image_dir="preprocess_images"):
    files = glob.glob(f"{image_dir}/*.jpg")
    latest_number = 0
    for file in files:
        image_num = int(file.replace(f"{image_dir}\\", "").replace(".jpg", ""))
        if image_num > latest_number:
            latest_number = image_num
    logging.info(f"latest number is {latest_number}")
    return latest_number


def get_video_paths(video_dir="videos"):
    return glob.glob(f"{video_dir}/*")


class FaceChecker:

    def __init__(self):
        self._face_cascade =  cv2.CascadeClassifier('ImageGenerator4StyleGan/haarcascade_frontalface_alt.xml')
        self._face_cascade1 = cv2.CascadeClassifier('ImageGenerator4StyleGan/haarcascade_frontalface_alt2.xml')
        self._face_cascade2 = cv2.CascadeClassifier('ImageGenerator4StyleGan/haarcascade_frontalface_alt_tree.xml')
        self._face_cascade3 = cv2.CascadeClassifier('ImageGenerator4StyleGan/haarcascade_frontalface_default.xml')
        self._eye_cascade =   cv2.CascadeClassifier('ImageGenerator4StyleGan/haarcascade_eye.xml')

        self._vision_client = vision.ImageAnnotatorClient()
        self.fallback_count = 0
        pass

    def has_face(self, frame) -> bool:

        # logging.info(f"check has face: size: {frame.shape}")
        # check has face using opencv
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        faces1 = self._face_cascade1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        faces2 = self._face_cascade2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        faces3 = self._face_cascade3.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        eyes = self._eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) != 0 or len(faces1) != 0 or len(faces2) != 0 or len(faces3) != 0 or len(eyes) != 0:
            return True

        self.fallback_count = self.fallback_count + 1
        logging.warning(f"fallback to google api: {self.fallback_count}")

        # check has face using google api
        cv2.imwrite("temp.jpg", frame)
        with io.open("temp.jpg", 'rb') as image_file:
            content = image_file.read()
        image = vision_v1.types.Image(content=content)
        response = self._vision_client.face_detection(image=image)
        if len(response.face_annotations) != 0:
            return True

        return False


def capture_image_from_mp4(destination_dir=None, interval=3):

    if not destination_dir:
        logging.error("capture destination directory must be specified, not None")
        exit(-1)

    # get video file names
    video_file_paths = get_video_paths()

    # get latest number
    frame_count = get_latest_number()
    save_count = 0

    # make instance check face
    face_checker = FaceChecker()

    # make directory
    make_directory(destination_dir)

    # main process
    for video_path in video_file_paths:
        print(f"tryin to open {video_path}")

        # make video player instance
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            exit(0)

        while True:
            ret, frame = cap.read()

            # check frame has data
            if ret:
                if save_count % interval == 0:
                    # NOTE: not recommend run face_checker.has_face frequently
                    if not face_checker.has_face(frame=frame):
                        # save image
                        path = f"{destination_dir}/{frame_count}.jpg"
                        cv2.imwrite(path, frame)

                        # increment frame count
                        frame_count = frame_count + 1
                    else:
                        continue
            else:
                """ almost cases is video is finished"""
                # TODO: handling error
                print(f"{video_path} process has finished!!")
                cap.release()
                break

            # increment save count
            save_count = save_count + 1


def renumbering(source_dir, destination_dir):
    """
    renumbering images and copy source dir to destination dir
    """
    make_directory(destination_dir)

    files = glob.glob(f"{source_dir}/*.jpg")
    latest_number = 0
    for file in files:
        print(f"renumbering {file} to renumber_images/{latest_number}.jpg")

        # load image
        im = cv2.imread(file)
        h, w, ch = im.shape

        # scale ratio
        ratio = 1.42

        # TODO: make scaled width is variable along image size
        scaled_width = 1920 * ratio
        scaled_margin_width = (scaled_width - 1024) / 2

        # transform setting
        m = np.float32([[1, 0, -600], [0, 1, -180]])

        # image process
        im_trasformed = cv2.warpAffine(im, m, (w, h))
        im_resized = cv2.resize(im_trasformed, dsize=None, fx=ratio, fy=ratio)
        dst = im_resized[0:1024, 0:1024]

        # save image
        cv2.imwrite(f"{destination_dir}/{latest_number}.jpg", im)

        latest_number = latest_number + 1


def main(only_renumbering=False):
    global SAVE_INTERVAL, IMAGE_ROOT_DIR, IMAGE_RESULT_DIR

    # capturing image
    if not only_renumbering:
        capture_image_from_mp4(
            destination_dir=IMAGE_ROOT_DIR,
            interval=SAVE_INTERVAL
        )

    # renumbering images after finishing process
    renumbering(
        source_dir=IMAGE_ROOT_DIR,
        destination_dir=IMAGE_RESULT_DIR
    )


if __name__ == '__main__':
    main(only_renumbering=True)
