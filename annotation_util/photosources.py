import os
import sys

import cv2


def validextension(filename):
    if '.jpg' in filename or '.jpeg' in filename or '.png' in filename:
        return True
    return False


class DirectorySource:
    def __init__(self, dirname):
        self.dirname = dirname
        print(f"\nDirectory '{dirname}' will be processed")
        self.list_of_valid_files = []
        for filename in [f.name for f in os.scandir(self.dirname) if (f.is_file() and validextension(f.name))]:
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
    def __init__(self, filename):
        self.filename = filename
        print(f"\nFile '{filename}' will be processed")
        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            print(f"Can not open {filename}! Abort...")
            exit(1)

    def next(self):
        ret, frame = self.cap.read()
        if ret:
            print("new frame")
            return None, frame
        else:
            raise StopIteration
