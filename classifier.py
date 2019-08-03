import anki_vector

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

import os
import tkinter as tk
from PIL import Image

from common import config
from control import connect_vector, control_vector
#from gui import ui

## use this function to load model
def main():
	## load model
	'''

	complete here !!!

	'''

	net = VGG('VGG19')
	checkpoint = torch.load('PrivateTest_model.t7', map_location='cpu')
	net.load_state_dict(checkpoint['net'])
	#net.cuda()
	net.eval()

	class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	# e.g. 
	# pretrained_dict = ...
	# model = torch.load(os.path.join(config.log_model_dir, ckpt_name))
	# model = torch.load('the-path-of-your-ckpt')

	def rgb2gray(rgb):
		return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
	
	def show_anim(robot, cnt):
		# nonlocal model

		# read the image sequence for classification
		img_path = config.img_path
		img_names = [os.path.join(img_path, 'face_round{}_cnt{}.jpg'.format(cnt, i)) for i in range(10)]
		face_seq = [io.imread(img) for img in img_names]

		anim_prob = np.zeros(len(config.anim_names))
		for face in face_seq:
			## use the model loaded to predict probabilities 
			## input one figure and get a one-dim array of size 7
			'''

			complete here !!!

			'''

			cut_size = 44

			transform_test = transforms.Compose([
    			transforms.TenCrop(cut_size),
    			transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
			])

			raw_img = face
			gray = rgb2gray(raw_img)
			gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

			img = gray[:, :, np.newaxis]

			img = np.concatenate((img, img, img), axis=2)
			img = Image.fromarray(img)
			inputs = transform_test(img)

			ncrops, c, h, w = np.shape(inputs)

			inputs = inputs.view(-1, c, h, w)
			#inputs = inputs.cuda()
			inputs = Variable(inputs, volatile=True)
			outputs = net(inputs)

			outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

			score = F.softmax(outputs_avg)
			_, predicted = torch.max(outputs_avg.data, 0)

			prob = score.data.cpu().numpy()

			anim_prob+=prob
			# e.g.
			# anim_prob += model(face)
		anim_prob = anim_prob / sum(anim_prob)

		anim_name = np.random.choice(config.anim_names, 1, p=anim_prob)[0]

		print('Playing Animation: ', anim_name)
		print('The Expression is: ', class_names[anim_prob.tolist().index(max(anim_prob))])
		robot.anim.play_animation(anim_name)


	# play the gui
	def ui():
		#parser = argparse.ArgumentParser()
		#parser.add_argument('-s', '--serial', dest='vector_serial', required=True)
		#parser.add_argument('-p', '--path', dest='score_path', required=True)
		#args = parser.parse_args()
	
		cnt = 0
		val = 0
	
		root = tk.Tk()
	
		robot = connect_vector(config.vector_serial)
	
		root.title('Vector Feedback')
		root.geometry('500x300')
	
		var1 = tk.StringVar()
		label1 = tk.Label(root, textvariable=var1, 
			bg='black', fg='white', font=('Arial', 12), width=30, height=2)
		# var1.set('How do you feel about vector?')
		label1.pack()
	
		var2 = tk.StringVar()
		label2 = tk.Label(root, textvariable=var2,
			bg='black', fg='white', font=('Arial', 8), width=5, height=1)
		var2.set('round:{}'.format(cnt))
		label2.pack()

		def start_test():
			nonlocal cnt
			nonlocal val
			control_vector(robot, cnt)
			show_anim(robot, cnt)
			cnt += 1
			val = 0
			var1.set('How do you feel about vector?')

		button2 = tk.Button(root, text='start',
			font=('Arial', 9), width=10, height=1, command=start_test)
		button2.pack()

		def end_test():
			nonlocal robot
			#robot.disconnect()
			root.destroy()

		button3 = tk.Button(root, text='end',
			font=('Arial', 9), width=10, height=1, command=end_test)
		button3.pack()

		def collect_score(v):
			nonlocal val
			val = v

		scale1 = tk.Scale(root, label='give a feedback', 
			from_=1, to=10, orient=tk.HORIZONTAL, length=200, 
			showvalue=1, tickinterval=1, resolution=1, command=collect_score)
		scale1.pack()

		def confirm_score():
			nonlocal cnt
			nonlocal val
			nonlocal robot
			if val != 0:
				var1.set('')
				with open(config.score_path, 'a') as f:
					f.write(str(cnt)+' '+str(val)+'\n')
				var2.set('round:{}'.format(cnt))
				control_vector(robot, cnt)
				show_anim(robot, cnt)
				cnt += 1
				val = 0
				var1.set('How do you feel about vector?')


		button1 = tk.Button(root, text='confirm', 
			font=('Arial', 9), width=10, height=1, command=confirm_score)
		button1.pack()
		
		root.mainloop()
		#robot.disconnect()

	ui()

if __name__ == '__main__':
	main()