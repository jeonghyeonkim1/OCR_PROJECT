# import cv2, os, urllib.request
# import numpy as np
# from django.conf import settings
# from imutils.video import VideoStream
# import imutils
# import pytesseract
# from imutils.object_detection import non_max_suppression
# import keyboard
# from PIL import ImageFont, ImageDraw, Image


# face_detection_videocam = cv2.CascadeClassifier(os.path.join(
#          cv2.data.haarcascades +'haarcascade_frontalface_default.xml'))
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# class VideoCamera(object):

#    def __init__(self):
#       self.video = cv2.VideoCapture(0)

#    def __del__(self):
#       self.video.release()

#    def get_frame(self):
      

#       if not self.video.isOpened():
#          self.video = cv2.VideoCapture(0)
#       if not self.video.isOpened():
#          raise IOError('cannot open webcam')

#       # net = cv2.dnn.readNet('./static/model/frozen_east_text_detection.pb')
   
#       cntr = 0
#       # 텍스트 ocr 처리
      
#       while True:
#          success, image = self.video.read()
         
			

#          cntr += 1
#          if((cntr%20) == 0):
#             imgh, imgw, chan = image.shape
#             x1,y1,w1,h1 = 0,0, imgh, imgw
#             imgchar = pytesseract.image_to_string(image, lang='kor')
#             imgboxes = pytesseract.image_to_boxes(image)

#             if keyboard.read_key() == 's':
#                cv2.imwrite()
#                print('제발저장')
            
#             # location = [
#             #    {
#             #          'idx': i,
#             #          'char': box.split(' ')[0],
#             #          'x': int(box.split(' ')[1]),
#             #          'y': int(box.split(' ')[2]),
#             #          'w': int(box.split(' ')[3]),
#             #          'h': int(box.split(' ')[4]),
#             #    }
#             #    for i, box in enumerate(imgboxes.splitlines()) if ord('가') <= ord(box.split(' ')[0]) <= ord('힣')
#             # ]
            
#             # temp_result = np.full((imgh, imgw, chan), fill_value=255, dtype=np.uint8)
#             # temp_result = Image.fromarray(temp_result)
#             # draw = ImageDraw.Draw(temp_result)

#             # for i in location:
#             #    cv2.rectangle(image, (i['x'], imgh-i['y']), (i['w'], imgh-i['h']), (0,0,0),3)
#             #    draw.text(
#             #       (i['x'] , i['y']), 
#             #       i['char'], 
#             #       font=ImageFont.truetype("fonts/gulim.ttc", 30),
#             #       fill=(255,0,0),
#             #    )

#             # temp_result = np.array(temp_result)


#             for boxes in imgboxes.splitlines():
#                boxes = boxes.split()
#                x1,y1,w1,h1 = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
#                cv2.rectangle(image, (x1, imgh-y1), (w1, imgh-h1), (0,0,0),3)
#                cv2.putText(image, boxes[0],(x1,imgh-h1+32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    
                    
#             # cv2.putText(image, imgchar, (x1+int(w1/50), y1+int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
            
            
            
#             # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             # faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#             # for (x, y, w, h) in faces_detected:
#             #    cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=1)
#             # # frame_flip = cv2.flip(image,-1)

				
#             _, jpeg = cv2.imencode('.jpg', image)
#             _, jpeg2 = cv2.imencode('.jpg', temp_result)
#             return jpeg.tobytes() , jpeg2.tobytes() 
      