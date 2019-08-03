import anki_vector

from common import config
#from show import show_anim
from collect import collect_img

def control_vector():
	args = anki_vector.util.parse_command_args()
	with anki_vector.Robot(args.serial) as robot:
		pass

def connect_vector(serial):
	try:
		robot = anki_vector.Robot(serial)
		robot.connect()
		return robot
	except Exception as e:
		print(e)

def control_vector(robot, cnt):
	# collect images
	collect_img(robot, cnt)
	# make face
	#show_anim(robot, cnt)