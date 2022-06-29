from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import numpy as np
import pytesseract
from django.views.decorators import gzip
import threading
# from myapp.camera import VideoCamera
from requests import Response
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from easyocr import Reader
import argparse
import cv2

def home(request):
    return render(request, 'home.html')

def scan(request):
    return render(request, 'base2.html')

class VideoCamera(object):
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        

    def __del__(self):
        self.video.release()

    def get_frame(self):
        
        while True:
            ret, image = self.video.read()
            if ret:
                # cv2.imshow('camera', image)
                if cv2.waitKey(1) == 13:
                    cv2.imwrite('photo.jpg', image)
                    a = cv2.imread('photo.jpg')
                    cv2.imshow('img', a)
            _, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def mycam(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass


