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


def ocr(request):
    context = {}
    if 'video_url' in request.POST:

        cap = cv2.VideoCapture(request.POST['video_url'])

        w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter('./OCR/static/out/output.avi',
                              fcc, fps, (w, h))

        while True:
            retval, frame = cap.read()
            if retval:
                out.write(frame)

            if not retval:
                break

            cv2.imshow('video_name', frame)

            if cv2.waitKey(50) == 27:
                break

        out.release()
        cap.release()

        cv2.destroyAllWindows()

    return render(request, 'base.html', context)


def video(request):
    context = {}

    model = r'C:\DevRoot\dataset\ComputerVision\dnnface\res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config = r'C:\DevRoot\dataset\ComputerVision\dnnface\deploy.prototxt'

    cap = cv2.VideoCapture('./OCR/static/out/output.avi')

    net = cv2.dnn.readNet(model, config)
    if net.empty():
        print('Net open failed!')
        exit()

    if not cap.isOpened():
        print('video open failed')
        sys.exit()

    while True:
        _, frame = cap.read()
        if frame is None:
            break

        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detect = net.forward()

        detect = detect[0, 0, :, :]
        (h, w) = frame.shape[:2]

        for i in range(detect.shape[0]):
            confidence = detect[i, 2]
            if confidence < 0.5:
                break

            x1 = int(detect[i, 3] * w)
            y1 = int(detect[i, 4] * h)
            x2 = int(detect[i, 5] * w)
            y2 = int(detect[i, 6] * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

            label = 'Face: %4.3f' % confidence
            cv2.putText(frame, label, (x1, y1 - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(50) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'base.html', context)
