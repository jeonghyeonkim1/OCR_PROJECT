from django.shortcuts import render
from functools import reduce
import pytesseract
import sys
import os
import cv2
import numpy as np
import pytesseract
import re
import time


def ocr_uploader(req):
    context = {}
    if 'uploadfile' in req.FILES:
        uploadfile = req.FILES.get('uploadfile', '')

        if uploadfile != '':
            cam = cv2.VideoCapture(uploadfile)
            fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            out = cv2.VideoWriter('/static/out/output.avi', fcc, fps, (width, height))

            while (cap.isOpened()):
                ret, frame = cap.read()
                w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

                if ret:
                    out.write(frame)
                    
                else:
                    print("Fail to read frame!")
                    break
                
            out.release()
            cap.release()

    return render(request, 'base.html', context)
    
def ocr(req):
    return render(request, 'base.html')
