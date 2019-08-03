import anki_vector
from anki_vector import events

import io
import sys
import os
import time
import numpy as np
from PIL import Image

from common import config
from util import img_aug

def collect_img(robot, cnt):
	'''collect images from vector's camera and save images

	Parameters
	----------
	robot: anki_vector.Robot object

	Returns
	-------
	'''
	img_path = config.img_path

	for i in range(10):
		time.sleep(0.2)
		raw_img = robot.camera.capture_single_image()
		#new_img = img_aug(robot, raw_img)
		#raw_img.raw_image.show()
		new_img = raw_img.raw_image

		save_img(new_img, img_path, cnt, i)

def save_img(img, img_path, cnt1, cnt2):
	'''save image to designated path

	Parameters
	----------
	img:
	img_path: str
		path to store 

	Returns
	-------
	'''
	img_name = 'face_round{}_cnt{}.jpg'.format(cnt1, cnt2)
	save_path = os.path.join(img_path, img_name)

	img.save(save_path, 'JPEG')