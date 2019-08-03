import os

class Config:
	# for pretrained model path
	log_dir = ''
	log_model_dir = ''
	#continue_path = ''

	img_path = './face_imgs_tst1'
	score_path = 'score_tst1.txt'

	vector_serial = '00702B42'
	anim_names = ['anim_eyepose_shocked', 'anim_eyepose_scared', 'anim_eyepose_concerned', 'anim_eyecontact_giggle_01_head_angle_40', 'anim_eyepose_sad', 'anim_eyepose_curious', 'anim_eyecontact_smile_01_head_angle_40']


config = Config()