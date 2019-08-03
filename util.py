import anki_vector

import numpy as np

from common import config

def img_aug(robot, face_img):
	'''preprocessing of face images

	Parameters
	----------
	robot: anki_vector.Robot object
	face_img: np.ndarray
		raw image from vector's camera (size [640, 360, 3])

	Returns
	-------
	new_img: np.ndarray
		image after augmentation
	'''
	# annotate face

	# rotation & scaling
	new_img = face_img.annotate_image(fit_size=(48, 48))
	
	#return new_img
	return face_img