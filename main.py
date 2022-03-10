# Import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
net = cv2.dnn.readNetFromCaffe('lib/deploy.prototxt', 'lib/res10_300x300_ssd_iter_140000.caffemodel')
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model('model.h5')

# load the input image
image = cv2.imread('Dataset/test data/1641494771535.jpg')
orig = image.copy()
(h, w) = image.shape[:2]
# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(124.0, 177.0, 123.0))
# pass the blob through the network and obtain the face detections
print("[INFO] computing face detections...")
net.setInput(blob)
faces = net.forward()

# loop over the detections
for i in range(0, faces.shape[2]):
	# Get face accuracy
	confidence = faces[0, 0, i, 2]
	# If face accuracy less than 0.5 go to the next face
	if confidence < 0.5:
		continue
	else:
		# Get boundary box of this face
		box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype('int')
		# Get face only form image
		face = image[startY:endY, startX:endX]
		# Preprocess this face for my own input
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		# Predict this face
		(mask, withoutMask) = model.predict(face)[0]
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# include the probability in the label
		label = f'{label}: {format(max((mask, withoutMask) * 100), ".3f")}%'
		# display the label and bounding box rectangle on the output
		cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
cv2.imshow("Output", image)
cv2.waitKey(0)
