
import cv2, os, urllib.request
import numpy as np
from django.conf import settings
from imutils.video import VideoStream
import imutils
import pytesseract

face_detection_videocam = cv2.CascadeClassifier(os.path.join(
			cv2.data.haarcascades +'haarcascade_frontalface_default.xml'))

class VideoCamera(object):

	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		font_scale = 1.5
		font = cv2.FONT_HERSHEY_PLAIN
		if not self.video.isOpened():
			raise IOError('cannot open webcam')
		cntr = 0
		while True:
			success, image = self.video.read()
			cntr += 1
			if((cntr%20) == 0):
				imgh, imgw, _ = image.shape
				x1,y1,w1,h1 = 0,0, imgh, imgw
				imgchar = pytesseract.image_to_string(image, lang='kor')
				imgboxes = pytesseract.image_to_boxes(image)

				for boxes in imgboxes.splitlines():
					boxes = boxes.split(' ')
					x1,y1,w1,h1 = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
					cv2.rectangle(image, (x1, imgh-y1), (w1, imgh-h1), (0,0,0),3)

				cv2.putText(image, imgchar, (x1+int(w1/50), y1+int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				
				
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
				for (x, y, w, h) in faces_detected:
					cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=1)
				# frame_flip = cv2.flip(image,-1)
				ret, jpeg = cv2.imencode('.jpg', image)
				return jpeg.tobytes()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		

		# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		# for (x, y, w, h) in faces_detected:
		# 	cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
		# frame_flip = cv2.flip(image,1)
		# ret, jpeg = cv2.imencode('.jpg', frame_flip)
			# return jpeg.tobytes()

