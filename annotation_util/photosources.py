import os
import sys

import cv2


def valid_photo_extension(filename):
    if '.jpg' in filename or '.jpeg' in filename or '.png' in filename:
        return True
    return False


def valid_video_extension(filename):
    if '.mp4' in filename or '.webm' in filename:
        return True
    return False


class PhotoDirectorySource:
    def __init__(self, dirname):
        self.dirname = dirname
        print(f"\nDirectory '{dirname}' will be processed")
        self.list_of_valid_files = []
        for filename in [f.name for f in os.scandir(self.dirname) if (f.is_file() and valid_photo_extension(f.name))]:
            self.list_of_valid_files.append(filename)
        self.list_position = 0

    def next(self):
        if self.list_position != len(self.list_of_valid_files):
            filepath = os.path.join(self.dirname, self.list_of_valid_files[self.list_position])
            print(filepath)
            self.list_position += 1
            return self.list_of_valid_files[self.list_position - 1], cv2.imread(filepath, cv2.IMREAD_COLOR)
        else:
            raise StopIteration


class VideoSource:
    def __init__(self, filename, strobe):
        self.filename = filename
        self.strobe = strobe
        print(f"\nFile '{filename}' will be processed")
        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            print(f"Can not open {filename}! Abort...")
            exit(1)

    def next(self):
        for i in range(self.strobe):
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
        print("new frame")
        return None, frame


class VideoDirectorySource:
    def __init__(self, dirname, strobe):
        self.strobe = strobe
        self.dirname = dirname
        print(f"\nDirectory '{dirname}' will be processed")
        self.list_of_valid_files = []
        for filename in [f.name for f in os.scandir(self.dirname) if (f.is_file() and valid_video_extension(f.name))]:
            self.list_of_valid_files.append(filename)
        self.list_position = 0
        filename = os.path.join(self.dirname, self.list_of_valid_files[self.list_position])
        print(f"\nFile '{filename}' will be processed")
        self.cap = cv2.VideoCapture(filename)
        self.list_position += 1

    def next(self):
        for i in range(self.strobe):
            ret, frame = self.cap.read()
            if not ret:
                if self.list_position == len(self.list_of_valid_files):
                    raise StopIteration
                else:
                    filename = os.path.join(self.dirname, self.list_of_valid_files[self.list_position])
                    print(f"\nFile '{filename}' will be processed")
                    self.cap = cv2.VideoCapture(filename)
                    self.list_position += 1
                    if not self.cap.isOpened():
                        print(f"Can not open {filename}! Abort...")
                        exit(1)
        print("new frame")
        return None, frame