# USAGE
# python detect_mask_video.py

# import the necessary packages
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# from imutils.video import VideoStream
from torchvision import transforms
from torch_mtcnn import detect_faces
from PIL import Image
import numpy as np
import argparse
# import imutils
import torch
import time
import cv2
import os

def detect_and_predict_mask(frame, maskNet, transform, device):
	# grab the dimensions of the frame and then construct a blob
	# from it
	# (h, w) = frame.shape[:2]
	# blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
	# 	(104.0, 177.0, 123.0))

    bbs, _ = detect_faces(img)

    faces = torch.tensor()
    predictions = []
    # bouding_boxes = []
    for i in range(0, len(bbs)):

        (top, left, bottom, right, _) = bbs[i].astype("int")

        img.crop((left, top, right, bottom))

        img = transform(img)

        faces = torch.cat((faces, img), dim=0)

        # bouding_boxes.append(bb)

    if len(faces) > 0:

        predictions = maskNet(faces.to(device))
        

    return (bbs, predictions)

if __name__ == '__main__':
	
		
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.7,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Let's use {}".format(device))

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")

	maskNet = torch.load(args["model"])
	maskNet.eval()

	transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = cv2.VideoCapture(0) #"http://192.168.0.3:4747/video") #VideoStream(src=0, framerate=16).start()  (1)
	# vs.open()
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		ret, frame = vs.read()
		# print(frame)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = Image.fromarray(img)

		# frame = cv2.resize(frame, (400, 400))
		# frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

		# frame = imutils.resize(frame, width=400)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, maskNet, transform, device)

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY, _) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
