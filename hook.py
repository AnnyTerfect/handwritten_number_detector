#-*- coding: utf8 -*-
import cv2
import numpy as np

H, W = 100, 100

def hook(model, img, flag):
	h, w, _ = img.shape
	_img = img

	a = min([h, w])
	img = 255 - np.average(img, axis=-1)
	img = img[h / 2 - H / 2: h / 2 + H / 2, w / 2 - W / 2: w / 2 + W / 2]
	img = cv2.resize(img, (28, 28))
	for i in range(28):
		for j in range(28):
				if img[i][j] > 127:
					img[i][j] = 255
				else:
					img[i][j] = 0

	if flag:
		cv2.imwrite('temp.png', img)

	img = img.reshape(1, 28, 28, 1)

	y = model.predict(img)
	y = np.argmax(y)

	font = cv2.FONT_HERSHEY_SIMPLEX
	_img[h / 2 - H / 2, w / 2 - W / 2: w / 2 + W / 2] = [255, 0, 0]
	_img[h / 2 + H / 2, w / 2 - W / 2: w / 2 + W / 2] = [255, 0, 0]
	_img[h / 2 - H / 2: h / 2 + H / 2, w / 2 - W / 2] = [255, 0, 0]
	_img[h / 2 - H / 2: h / 2 + H / 2, w / 2 + W / 2] = [255, 0, 0]
	_img = cv2.putText(_img, 'The number is:' + str(y), (50, 50), font, 1.2, (255, 0, 0), 4)

	return _img