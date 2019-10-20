import cv2
import numpy as np
from keras.models import load_model

img = cv2.imread("130.jpg",0)
img = cv2.resize(img, (160,96))
img=np.float32(img)
img/=255.0
img = img.reshape(1,96,160,1)
model = load_model("temp.h5")

img = model.predict(img)

print(img)
#img=img.reshape(96,160,3)
cv2.imshow("image",img[0])
cv2.waitKey(0)
