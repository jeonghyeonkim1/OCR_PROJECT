from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import pytesseract
from django.views.decorators import gzip
import threading
from myapp.camera import VideoCamera
from requests import Response


def scan(request):
    return render(request, 'base2.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def mycam(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")

