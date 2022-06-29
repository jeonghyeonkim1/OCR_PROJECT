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
        

        cntr = 0
        while True:
            _, image = self.video.read()
            # cntr += 1
            # if((cntr%20) == 0):
            #     imgh, imgw, chan = image.shape
                # x1,y1,w1,h1 = 0,0,imgh,imgw
                # imgchar = pytesseract.image_to_string(image, lang='kor')
                # imgboxes = pytesseract.image_to_boxes(image)

                # for boxes in imgboxes.splitlines():
                #     boxes = boxes.split()
                #     x1,y1,w1,h1 = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
                #     cv2.rectangle(image, (x1, imgh-y1), (w1, imgh-h1), (0,0,0),3)
                #     cv2.putText(image, boxes[0],(x1,imgh-h1+32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    
                
                # (self.grabbed, self.frame) = self.video.read()
                # threading.Thread(target=self.update, args=()).start()
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

# def gen(camera, camera2):
#     while True:
#         frame = camera.get_frame()
#         frame2 = camera2.get_frame()
#         yield(b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         yield(b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')


# def mycam(request):
#     return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")

