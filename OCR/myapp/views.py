from ast import While
from multiprocessing.connection import wait
from turtle import window_height, window_width
from django.shortcuts import render
from django.http import JsonResponse
import os
import time
import cv2
import numpy as np
import pytesseract
# from django.views.decorators import gzip
import threading
# from myapp.camera import VideoCamera
from requests import Response
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from easyocr import Reader
import argparse
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import roboflow
from roboflow import Roboflow
from tkinter import *

root = Tk()

monitor_height = root.winfo_screenheight()
monitor_width = root.winfo_screenwidth()

rf = Roboflow(api_key="BzyHkzKOlMSJcspr3EH2")
workspace = rf.workspace()
workspace.name
workspace.url
workspace.projects()

project = rf.workspace("seoheejang").project("chess-sample-m5hzq")
model = project.version(6).model

def home(request):
    return render(request, 'home.html')

def scan(request):
    # args = {
    #         "image": "./static/contour_list/contour.jpg",
    #         "langs": "en,ko",
    #         "gpu": -1
    #     }
    # langs = args["langs"].split(",")
    # image = cv2.imread('./static/contour_list/contour.jpg')
    # reader = Reader(langs, gpu=args["gpu"] > 0)
    # results = reader.readtext(image, )
    # for (bbox, text, prob) in results:
    #     # display the OCR'd text and associated probability
    #     print("[INFO] {:.4f}: {}".format(prob, text))

    #     # unpack the bounding box
    #     (tl, tr, br, bl) = bbox
    #     tl = (int(tl[0]), int(tl[1]))
    #     tr = (int(tr[0]), int(tr[1]))
    #     br = (int(br[0]), int(br[1]))
    #     bl = (int(bl[0]), int(bl[1]))

    #     # cleanup the text and draw the box surrounding the text along
    #     # with the OCR'd text itself
    #     text = cleanup_text(text)
    #     cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    #     cv2.putText(image, text, (tl[0], tl[1] - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.imwrite('./static/contour_list/textocr_result.jpg', image)
    # cv2.destroyAllWindows()
    return render(request, 'result.html')

def get_cam(request):
    file_path = './static/contour_list'
    os.mkdir(file_path) if not os.path.exists(file_path) else ''

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print('camera open failed') if not capture.isOpened() else ''

    # cv2.resizeWindow('qwe', monitor_width, monitor_height)
    # cv2.moveWindow('qwe', 0, 0)

    while 1:
        ret, frame = capture.read()
        if not ret:
            break

        cv2.imwrite('./static/data/hi.jpg', frame)

        prediction = model.predict('./static/data/hi.jpg').json()['predictions']

        if cv2.waitKey(1) == 27:
            capture.release() 
            cv2.destroyAllWindows()
            return JsonResponse({"false"})
        
        if len(prediction) > 0:
            predict = prediction[0]

            if predict['class'] == 'book' and predict['confidence'] >= 0.1:
                x, y, w, h = int(predict['x']), int(predict['y']), int(predict['width']), int(predict['height'])

                cv2.rectangle(
                    frame,
                    pt1=(x - int(w/2), y - int(h/2)),
                    pt2=(x + int(w/2), y + int(h/2)),
                    color=(0, 255, 255),
                    thickness=2
                )

                cv2.putText(
                    frame,
                    text = f"{predict['class']} {round(predict['confidence'] * 100, 1)}%",
                    org = (x - int(w/2), y - int(h/2) - 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color = (0, 255, 255),
                    thickness=2
                )

        cv2.imshow('qwe', frame)

        if cv2.waitKey(1) == 13:
            if len(prediction) > 0 and predict['class'] == 'book' and predict['confidence'] >= 0.1:
                contour = {
                    'lu': (x - int(w/2), y - int(h/2)),
                    'ru': (x + int(w/2), y - int(h/2)),
                    'rb': (x + int(w/2), y + int(h/2)),
                    'lb': (x - int(w/2), y + int(h/2)),
                    'width': predict['width'],
                    'height': predict['height'],
                }

                srcQuad = np.array([
                    contour['lu'], contour['ru'], 
                    contour['rb'], contour['lb'],
                ]).astype(np.float32)

                dstQuad = np.array([
                    [0, 0], [contour['width'], 0],
                    [contour['width'], contour['height']], [0, contour['height']],
                ]).astype(np.float32)

                pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
                dst = cv2.warpPerspective(frame, pers, (w, h))

                cv2.imwrite('./static/contour_list/contour.jpg', dst)
                break
                
            else:
                capture.release() 
                cv2.destroyAllWindows()
                return JsonResponse({'notCaptured': "캡쳐되지 않았습니다!"})

    capture.release() 
    cv2.destroyAllWindows()
    return JsonResponse({})


def recommend(request):
    return render(request, 'recommend.html')


def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def loading(request):
    return render(request, 'loading.html')