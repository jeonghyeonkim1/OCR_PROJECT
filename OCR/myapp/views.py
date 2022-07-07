from django.shortcuts import render
from django.http import JsonResponse
import os
import cv2
from matplotlib.pyplot import text
import numpy as np
from easyocr import Reader
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from roboflow import Roboflow
import requests
from bs4 import BeautifulSoup

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
    

    img = cv2.imread('./static/contour_list/contour.jpg')

    

    reader = Reader(['en', 'ko'], gpu=False)
    
    

    results = reader.readtext(img)


    text_list = []
    for (bbox, text, prob) in results:
        print("[INFO] {:.4f}: {}".format(prob, text))

        (tl, tr, br, bl) = bbox

        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        text_list.append(text)

        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        
        print(text)

        cv2.rectangle(
            img,
            pt1 = tl,
            pt2 = br,
            color = (0, 255, 0),
            thickness = 2
        )

    
    cv2.imwrite('./static/contour_list/text_contour.jpg', img)
    
    return render(request, 'result.html', {'text': ' '.join(text_list)})

def get_cam(request):
    file_path = './static/contour_list'
    os.mkdir(file_path) if not os.path.exists(file_path) else ''

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print('camera open failed') if not capture.isOpened() else ''

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


def book_search(search):
    url = f'https://play.google.com/store/search?q={search}&c=books'

    headers = {
        'referer': 'https://play.google.com/',
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers, stream=True)

    dom = BeautifulSoup(response.text, 'html.parser')

    rolelist = dom.select('[role="listitem"]')
    reply = dom.select('p a')

    if len(reply) > 0:
        url = f'https://play.google.com/store/search?q={reply[0].text.strip()}&c=books'

        response = requests.get(url, headers=headers, stream=True)

        dom = BeautifulSoup(response.text, 'html.parser')

        rolelist = dom.select('[role="listitem"]')
        
        
    result = []

    for book_list in rolelist:


        title = book_list.select_one('div img').attrs['alt'] if book_list.select_one('div img') != None else ''


        thumbnail = book_list.select_one('div img').attrs['src'] if book_list.select_one('div img') != None else ''

        if title != '' and thumbnail != '':
            result.append({
                'title': title,
                'thumbnail': thumbnail,
            })

    return result


def recommend(request):
    context = {
        'text' : request.GET.get('text_2')
        }
    
    
    print(context['text'])
    return render(request, 'recommend.html', context )

def loading(request):
    return render(request, 'loading.html')


