from django.shortcuts import render
from django.http import JsonResponse
import os
import cv2


def enter(request):
    return render(request, 'base.html')


def enter_call(request):
    if 'video_url' in request.GET:

        # 모델 선언
        model = './static/ComputerVision/dnnface/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        config = './static/ComputerVision/dnnface/deploy.prototxt'
        net = cv2.dnn.readNet(model, config)
        if net.empty():
            return JsonResponse({'error': '모델이 업솨영'})

        # 영상로더 선언
        cap = cv2.VideoCapture(request.GET['video_url'])
        if not cap.isOpened():
            return JsonResponse({'error': '영상이 업솨영'})

        # Frame load
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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

            cv2.imshow('Count People', frame)

            if cv2.waitKey(1) == 27:  # esc를 누르면 강제 종료
                break

        cap.release()
        cv2.destroyAllWindows()

        return JsonResponse({'success': '영상 업로드 완료!'})
