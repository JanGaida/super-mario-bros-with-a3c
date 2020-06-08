""" © Jan Gaida, 2020 """

# Generel
import numpy as np
import cv2, colorsys

# Gym
import gym
from gym import Wrapper
from gym.spaces import Box
from gym.wrappers import Monitor

# Super-Mario-Bros-Gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


""" enviorment.py

Definiert das Super-Mario-Bors-Enviorment (basierend auf https://github.com/Kautenja/gym-super-mario-bros)


- PreprocessFrameWrapper:
Enviorment-Wrapper zum Preprocessing des Frames

- RewardWrapper:
Enviorment-Wrapper zum bestimmen des Rewards

- FrameBufferWrapper:
Enviorment-Wrapper zum sammeln einer Framereihenfolge

- make_training_enviorment:
Hilfsfunktion zum erstellen eines Training-Enviorments

- make_testing_enviorment:
Hilfsfunktion zum erstellen eines Testing-Enviorments

"""


np.seterr(over = 'ignore') # Ignoriere Numpy-Warnings
gym.logger.set_level(40) # Gym-Logger-Level -> DEBUG = 10  INFO = 20  WARN = 30  ERROR = 40  DISABLED = 50
enviorment_out_width = 64
enviorment_out_height = 50


class PreprocessFrameWrapper(Wrapper):
	"""Bearbeitet alle Frames des Enviorments"""

	def __init__(self, env):
		super(PreprocessFrameWrapper, self).__init__(env)
		"""Init"""
		# Zähler für das herausschreiben von Frames
		self.counter = -1 
		
	def step(self, action):
		"""leite den Step weiter"""
		state, reward, done, info = self.env.step(action) # den Step auffangen
		return self.preprocess_frame(state), reward, done, info # den Frame verarbeiten

	def reset(self):
		"""leitet den Reset-Call weiter"""
		return self.preprocess_frame(self.env.reset()) # Leite den call weiter und verarbeite den Frame

	def preprocess_frame(self, frame):
		"""Vereinfacht das übergebe Frame"""
		global enviorment_out_width
		global enviorment_out_height

		# Orginale Shape: (240, 256, 3)
		if frame is not None:
			""" Um Frames herauszuschreiben
			if self.counter <= 1000:
				self.counter += 1
				if self.counter % 50 == 0:
					frame_ = frame[15:215,:]
					cv2.imwrite("img/cut_{}.jpg".format(self.counter), frame_)
					_, _ , frame_ = cv2.split(frame)
					cv2.imwrite("img/black_n_white_{}.jpg".format(self.counter), frame_)
					_, frame_ = cv2.threshold(frame_, 64, 255, cv2.THRESH_TOZERO)
					_, frame_ = cv2.threshold(frame_, 64, 255, cv2.THRESH_BINARY_INV)
					cv2.imwrite("img/threshold_{}.jpg".format(self.counter), frame_)
					frame_ = cv2.resize(frame_, (enviorment_out_width, enviorment_out_height), interpolation=cv2.INTER_AREA)
					cv2.imwrite("img/resized_{}.jpg".format(self.counter), frame_)
			""" 

			# In Schwarz-Weißen-Ausschnitt umwandeln
			_, _ , frame = cv2.split( frame[15:215,:] )
			# Threshold anwenden
			_, frame = cv2.threshold(frame, 64, 255, cv2.THRESH_TOZERO)
			_, frame = cv2.threshold(frame, 64, 255, cv2.THRESH_BINARY_INV)
			# Shape anpassen
			return (cv2.resize(frame, (enviorment_out_width, enviorment_out_height), interpolation=cv2.INTER_AREA))[None, :, :] / 255.

		# Ansonsten
		else:
			# Leeres Bild
			return np.zeros((1, enviorment_out_height, enviorment_out_width))


class RewardWrapper(Wrapper):
	"""Überschreibt die Reward-Funktion
		Info-Dict ~> https://github.com/Kautenja/gym-super-mario-bros#info-dictionary
		Org. Reward ~> https://github.com/Kautenja/gym-super-mario-bros#reward-function
	"""

	def __init__(self, env):
		super(RewardWrapper, self).__init__(env)
		"""Init"""
		self.x_0 = 40 # Mario's initiale X-Position
		self.score_0 = 0 # Mario's initialer Score
		self.clock_0 = 0 # Mario's initiale Zeit, Anmerkung: ggf. 300

	def step(self, action):
		"""Leitet den Step call weiter und übschreibt den Reward"""
		# leite den Step weiter
		state, reward, done, info = self.env.step(action)

		# Var's lesen
		x_1 = info['x_pos']
		score_1 = info['score']
		clock_1 = info['time']

		reward =  ( max( x_1 - self.x_0, -5 ) ) \
				+ ( max( score_1 - self.score_0, 0 ) / 400. ) \
				+ ( clock_1 - self.clock_0 ) / 10. \
				+ ( 0. if not done else 50. if info['flag_get'] else -25.)


		""" 26min w1s1
		reward =  ( max( x_1 - self.x_0, -5 ) ) \
				+ ( max( score_1 - self.score_0, 0 ) / 400. ) \
				+ ( clock_1 - self.clock_0 ) / 10. \
				+ ( 0. if not done else  35. if info['flag_get'] else -35.) \
				+ ( max( life_1 - self.life_0, 0 ) * 35.)
		"""

		"""        
		reward =  ( max( x_1 - self.x_0, -100) ) \
				+ ( max( score_1 - self.score_0, 0 ) / 400. ) \
				+ ( min( clock_1 - self.clock_0, 0 ) / 10. ) \
				+ ( 0. if not done else  35. if info['flag_get'] else -35.) #\
				#+ ( -10. if life_1 < self.life_0 else 0. )
		"""


		"""w/o w1s3
		reward =  ( max( x_1 - self.x_0, -5 ) ) \
				+ ( max( score_1 - self.score_0, 0 ) / 400. ) \
				+ ( clock_1 - self.clock_0 ) / 10. \
				+ ( 0. if not done else  50. if info['flag_get'] else -50.) \
				+ ( -40. if not life_1 == self.life_0 else 0. )
		"""

		"""reward =  ( (x_1 - self.x_0) / 4. ) \
				+ ( max(clock_1 - self.clock_0, -1) ) \
				+ ( max( score_1 - self.score_0, 0 ) / 400. ) \
				+ ( 0. if not done else  50. if info['flag_get'] else -50.) \
				+ ( -75. if not life_1 == self.life_0 else 0. )"""

		# VARs updaten
		self.x_0 = x_1
		self.score_0 = score_1
		self.clock_0 = clock_1

		# Fertig
		return state, reward / 10. , done, info

	def reset(self):
		"""Leitet den Reset-Call weiter"""
		self.x_0 = 40
		self.score_0 = 0
		self.clock_0 = 0

		# Weiterleiten
		return self.env.reset()

	def status_to_int(self, status):
		"""Hilfsfunktion um den Status von Mario in einen vergleichbaren Integer zu wandeln"""
		return 1 if not status == "small" else 0


class FrameBufferWrapper(Wrapper):
	"""Puffert einige Frames in einem Numpy-Array, überschreibt den Observation_Space dementsprechend"""

	def __init__(self, env, skip):
		super(FrameBufferWrapper, self).__init__(env)
		"""Init"""
		global enviorment_out_width
		global enviorment_out_height

		# Merk wie viel Frames übersprungen werden sollen
		self.skip = skip

		# Überschreib den Observation_Space
		self.observation_space = Box(low = 0, high = 255, shape = (skip, enviorment_out_height, enviorment_out_width), dtype = np.float32)

	def step(self, action):
		"""Leite den Step-Call weiter und puffert die Frames"""
		# zusammelnde Vars
		states = []
		sum_reward = 0

		# Frames buffern
		state, reward, done, info = self.env.step(action) # 0. Step
		for _ in range(self.skip): # 1. bis n. Step
			# Wenn Env nicht abgeschlossen wurde
			if not done:
				state, reward, done, info = self.env.step(action) # .. mach den Step
				sum_reward += reward # .. bilde die Summe des Rewards
				states.append(state) # .. füg den State hinzu

			# Wenn Env abgeschlossen wurde
			else:
				states.append(state) # .. fülle mit den letzten State auf

		# Aus den gesammelten States ein Array bauen
		states = np.concatenate(states, 0)[None, :, :, :]

		# Float-Genauigkeit reduzieren
		return states.astype(np.float32), reward, done, info

	def reset(self):
		"""Leitet den Reset-Call weiter"""
		# Weiterleiten
		state = self.env.reset()

		# State wiederholen und zu Array umformen
		states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]

		# Genauigkeit reduzieren
		return states.astype(np.float32)


def make_training_enviorment(args):
	"""Erzeugt ein Trainings-Enviorment"""
	# Gym
	env = gym.make("SuperMarioBros-{}-{}-v{}".format(args.world, args.stage, args.rversion))

	# JoypadSpace-Wrapper, vgl. https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
	if args.action_set == "rightonly":
		action_set = RIGHT_ONLY
	elif args.action_set == "simple":
		action_set = SIMPLE_MOVEMENT
	elif args.action_set == "complex":
		action_set = COMPLEX_MOVEMENT
	else:
		raise Exception("Invalde Actions.")
	env = JoypadSpace(env, action_set)

	# Berarbeiten der Frames
	env = PreprocessFrameWrapper(env)

	# Überschreiben des Rewards
	env = RewardWrapper(env)

	# Überspringen von Frames
	env = FrameBufferWrapper(env, args.skip_frames)

	# Rückgabe
	num_states = env.observation_space.shape[0]
	num_actions = len(action_set)

	return env, num_states, num_actions


def make_testing_enviorment(args, episode):
	"""Erzeugt ein Testing-Enviorment"""

	# Gym
	env = gym.make("SuperMarioBros-{}-{}-v{}".format(args.world, args.stage, args.rversion))

	# JoypadSpace-Wrapper, vgl. https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
	if args.action_set == "rightonly":
		action_set = RIGHT_ONLY
	elif args.action_set == "simple":
		action_set = SIMPLE_MOVEMENT
	elif args.action_set == "complex":
		action_set = COMPLEX_MOVEMENT
	else:
		raise Exception("Invalde Actions.")
	env = JoypadSpace(env, action_set)

	# Wende Monitor an
	env = Monitor(env, "{}/a3c_smb_world{}_stage{}_ver{}/ep{}".format(args.recordsdir, args.world, args.stage, args.rversion, episode), force=True)

	# Berarbeiten der Frames
	env = PreprocessFrameWrapper(env)

	# Überschreiben des Rewards
	env = RewardWrapper(env)

	# Überspringen von Frames
	env = FrameBufferWrapper(env, args.skip_frames)

	# Rückgabe
	num_states = env.observation_space.shape[0]
	num_actions = len(action_set)

	return env, num_states, num_actions
