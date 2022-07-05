from ast import While
from multiprocessing.connection import wait
from django.shortcuts import render
from django.http import JsonResponse
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

res = {}

def home(request):
    return render(request, 'home.html')

def scan(request):  
    return render(request, 'base2.html')

def get_cam(request):
    capture = cv2.VideoCapture(0)
    print('camera open failed') if not capture.isOpened() else ''

    while 1:
        ret, frame = capture.read()
        if not ret:
            break

        cv2.imshow('qwe', frame)
        
        if cv2.waitKey(1) == 13:
            break

    capture.release() 
    cv2.destroyAllWindows()
    return JsonResponse({})


def mycam(request):  
    camera = cv2.VideoCapture(0)
    print('camera open failed') if not camera.isOpened() else ''
    # camera = cv2.VideoCapture(cv2.CAP_DSHOW+0)

    rf = Roboflow(api_key="BzyHkzKOlMSJcspr3EH2")
    workspace = rf.workspace()
    workspace.name
    workspace.url
    workspace.projects()

    project = rf.workspace("seoheejang").project("chess-sample-m5hzq")
    model = project.version(6).model


    
    ret , img = camera.read()

    # if not ret: break
    # prediction = model.predict(img)
    # print(prediction)
    cv2.imshow('sdf', img)
    if cv2.waitKey(0) == 27:
        pass
        # break 
        
    camera.release()
    cv2.destroyAllWindows()
    return JsonResponse({})

def cam(request):
    camera = cv2.VideoCapture(cv2.CAP_DSHOW+0)

    while True:
        ret, image = camera.read()
        if ret:
            bbox, label, conf = cv.detect_common_objects(image)
            if 'book' in label:
                lab_ind = label.index('book')
                bbox=[bbox[lab_ind]]
                label=[label[lab_ind]]
                conf=[conf[lab_ind]]
                if conf[0] >= 0.7:
                    cv2.imwrite('./static/data/photo.jpg', image)
                    cv2.destroyAllWindows()
                    def cleanup_text(text):
                        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

                    args = {
                        "image": "./static/data/photo.jpg",
                        "langs": "ko,en",
                        "gpu": -1
                    }

                    langs = args["langs"].split(",")
                    print("[INFO] OCR'ing with the following languages: {}".format(langs))

                    image = cv2.imread(args["image"])


                    print("[INFO] OCR'ing input image...")
                    reader = Reader(langs, gpu=args["gpu"] > 0)
                    results = reader.readtext(image)

                    for (bbox, text, prob) in results:
                        print("[INFO] {:.4f}: {}".format(prob, text))

                        (tl, tr, br, bl) = bbox
                        tl = (int(tl[0]), int(tl[1]))
                        tr = (int(tr[0]), int(tr[1]))
                        br = (int(br[0]), int(br[1]))
                        bl = (int(bl[0]), int(bl[1]))

                        text = cleanup_text(text)
                        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                        cv2.putText(image, text, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        cv2.imwrite('./static/data/ocr_result.jpg',image)
                        cv2.destroyAllWindows()

                        res = text
                        return JsonResponse({'data': res})            
            else:
                bbox=[]
                label=[]
                conf=[]

            out = draw_bbox(image, bbox, label, conf, write_conf=True)
            cv2.imshow('Real-time object detection', out)
        
    camera.release()
    cv2.destroyAllWindows()
    return JsonResponse({})


def recommend(request):
    content = {
        'data':res,
    }
    return render(request, 'recommend.html',content)


