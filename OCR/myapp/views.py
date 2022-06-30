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
                cv2.imshow('camera', image)
                if cv2.waitKey(1) == 13:
                    cv2.imwrite('./static/photo.jpg', image)
                    a = cv2.imread('./static/photo.jpg')
                    cv2.imshow('img', a)
                    if cv2.waitKey(0) == 13:

                        def cleanup_text(text):
                            # strip out non-ASCII text so we can draw the text on the image
                            # using OpenCV
                            return "".join([c if ord(c) < 128 else "" for c in text]).strip()

                        # construct the argument parser and parse the arguments
                        #ap = argparse.ArgumentParser()
                        #ap.add_argument("-i", "--image", required=True,
                        #   help="path to input image to be OCR'd")
                        #ap.add_argument("-l", "--langs", type=str, default="en",
                        #   help="comma separated list of languages to OCR")
                        #ap.add_argument("-g", "--gpu", type=int, default=-1,
                        #   help="whether or not GPU should be used")
                        #args = vars(ap.parse_args())

                        # since we are using Jupyter Notebooks we can replace our argument
                        # parsing code with *hard coded* arguments and values
                        args = {
                            "image": "./static/photo.jpg",
                            "langs": "ko,en",
                            "gpu": -1
                        }

                        # break the input languages into a comma separated list
                        langs = args["langs"].split(",")
                        print("[INFO] OCR'ing with the following languages: {}".format(langs))

                        # load the input image from disk
                        image = cv2.imread(args["image"])

                        # OCR the input image using EasyOCR
                        print("[INFO] OCR'ing input image...")
                        reader = Reader(langs, gpu=args["gpu"] > 0)
                        results = reader.readtext(image)

                        # loop over the results
                        for (bbox, text, prob) in results:
                            # display the OCR'd text and associated probability
                            print("[INFO] {:.4f}: {}".format(prob, text))

                            # unpack the bounding box
                            (tl, tr, br, bl) = bbox
                            tl = (int(tl[0]), int(tl[1]))
                            tr = (int(tr[0]), int(tr[1]))
                            br = (int(br[0]), int(br[1]))
                            bl = (int(bl[0]), int(bl[1]))

                            # cleanup the text and draw the box surrounding the text along
                            # with the OCR'd text itself
                            text = cleanup_text(text)
                            cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                            cv2.putText(image, text, (tl[0], tl[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        cv2.imwrite('./static/photo1.jpg',image)
                        cv2.destroyAllWindows()
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

