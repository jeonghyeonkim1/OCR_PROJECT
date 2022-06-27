from django.shortcuts import render
from django.http import JsonResponse
import os
import cv2
import numpy as np


def enter(request):
    return render(request, 'base.html')


def enter_call(request):
    # 파라미터
    CONFIDENCE = 0.9
    THRESHOLD = 0.3
    LABELS = ['Car', 'Plate']
    CAR_WIDTH_TRESHOLD = 500

    # 영상로더 선언
    cap = cv2.VideoCapture(request.GET['video_url'])
    if not cap.isOpened():
        cap = cv2.VideoCapture('C:/OCR_DATA/car.mp4')
        if not cap.isOpened():
            return JsonResponse({'error': '영상이 업솨영'})

    # 모델 선언
    net = cv2.dnn.readNetFromDarknet(
        './mycar/cfg/yolov4-ANPR.cfg', 'C:/OCR_DATA/yolov4-ANPR.weights')
    if net.empty():
        return JsonResponse({'error': '모델이 업솨영'})

    # Frame load
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

        H, W, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255., size=(416, 416), swapRB=True)
        net.setInput(blob)
        output = net.forward()

        boxes, confidences, class_ids = [], [], []

        for det in output:
            box = det[:4]
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE:
                cx, cy, w, h = box * np.array([W, H, W, H])
                x = cx - (w / 2)
                y = cy - (h / 2)

                boxes.append([int(x), int(y), int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        print(idxs)

        if len(idxs) > 0:
            for i in sorted(idxs.flatten()):
                x, y, w, h = boxes[i]

                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h),
                            color=(0, 0, 255), thickness=2)
                cv2.putText(frame, text='%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), org=(
                    x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

                print(x, y, w, h)

                if class_ids[i] == 0:
                    if w > CAR_WIDTH_TRESHOLD:
                        pass
                    else:
                        pass
        else:
            pass

        print("-" * 40)

        cv2.imshow('Count People', frame)

        if cv2.waitKey(1) == 27:  # esc를 누르면 강제 종료
            break

    cap.release()
    cv2.destroyAllWindows()

    return JsonResponse({'success': '영상 업로드 완료!'})
