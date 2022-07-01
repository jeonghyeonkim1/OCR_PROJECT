from multiprocessing.connection import wait
from django.shortcuts import render
from django.http import JsonResponse
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
import cvlib as cv
from cvlib.object_detection import draw_bbox


def home(request):
    return render(request, 'home.html')

def scan(request):            
    return render(request, 'base2.html')

def cam(request):
    camera = cv2.VideoCapture(0)

    while True:
        ret, image = camera.read()


        if ret:
            bbox, label, conf = cv.detect_common_objects(image)
            if 'book' in label:
                lab_ind = label.index('book')
                bbox=[bbox[lab_ind]]
                label=[label[lab_ind]]
                conf=[conf[lab_ind]]
                if 0.6 >= conf[0] >= 0.5:
                    cv2.imwrite('./static/photo.jpg', image)
                    # a = cv2.imread('./static/photo.jpg')
                    # cv2.imshow('img', a)
                    if cv2.waitKey(0) == 13:
                        def cleanup_text(text):
                            return "".join([c if ord(c) < 128 else "" for c in text]).strip()


                        args = {
                            "image": "./static/photo.jpg",
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

                        cv2.imwrite('./static/ocr_result.jpg',image)
                        cv2.destroyAllWindows()

                        return JsonResponse({'img_path1': './static/photo.jpg', 'img_path2': './static/ocr_result.jpg'})            
            else:
                bbox=[]
                label=[]
                conf=[]

            out = draw_bbox(image, bbox, label, conf, write_conf=True)
            cv2.imshow('Real-time object detection', out)
            cv2.waitKey(1)
            if cv2.waitKey(1) == 27:
                break
    cv2.release()
    cv2.destroyAllWindows()
    return JsonResponse({'img_path1': './static/photo.jpg', 'img_path2': './static/ocr_result.jpg'})

