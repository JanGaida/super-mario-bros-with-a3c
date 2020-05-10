# Generel
import numpy as np
import cv2
import subprocess as sp

# Gym
import gym
from gym import Wrapper
from gym.spaces import Box

# Super-Mario-Bros-Gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

# Ignoriere Numpy-Warnings -> https://github.com/RunzheYang/MORL/issues/5
np.seterr(over = 'ignore')

# Gym-Logger-Level -> DEBUG = 10  INFO = 20  WARN = 30  ERROR = 40  DISABLED = 50
gym.logger.set_level(40)

class PreprocessFrameWrapper(Wrapper):
    """Bearbeitet alle Frames des Enviorments"""

    def __init__(self, env):
        super(PreprocessFrameWrapper, self).__init__(env)

    def step(self, action):
        # leite den Step weiter
        state, reward, done, info = self.env.step(action)
        return preprocess_frame(state), reward, done, info

    def reset(self):
        # leite den Reset weiter & bearbeite diesen
        return preprocess_frame(self.env.reset())


class RewardWrapper(Wrapper):
    """Überschreibt die Reward-Funktion"""

    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)
        self.score_0 = 0
        self.coin_0 = 0

    def step(self, action):
        # Info-Dict ~> https://github.com/Kautenja/gym-super-mario-bros#info-dictionary
        # Org. Reward ~> https://github.com/Kautenja/gym-super-mario-bros#reward-function

        # leite den Step weiter
        state, reward, done, info = self.env.step(action)

        # hier berrechnet Reward liegt zwischen -15 und 15 
        # reward = (x_pos_1 - x_pos_0) + (clock_0 - clock_1) + (alive ? 0 ansonsten -15)

        # Score:
        score_1 = info["score"]
        my_reward = (score_1 - self.score_0) * .01

        # Coins
        coin_1 = info["coins"]
        my_reward = (coin_1 - self.coin_0) * .1

        self.score_0 = score_1
        self.coin_0 = coin_1

        # Wenn das Enviorment abgeschlossen ist
        if done:
            # Und das Ziel erreicht wurde
            if info["flag_get"]:
                my_reward += 45

            # Und das Ziel _nicht_ erreicht wurde
            else:
                my_reward -= 45

        reward += my_reward * .1

        return state, reward, done, info

    def reset(self):
        # Letzten Score ebenfalls zurücksetzten
        self.score_0 = 0
        self.coin_0 = 0

        # Weiterleiten
        return self.env.reset()


class SkipFrameWrapper(Wrapper):

    def __init__(self, env, skip):
        super(SkipFrameWrapper, self).__init__(env)
        # Überschreib den Observation_Space
        self.observation_space = Box(low = 0, high = 255, shape = (4, 84, 84), dtype = np.float32)
        # Merk wie viel Frames übersprungen werden sollen
        self.skip = skip - 1

    def step(self, action):
        # zusammelnde Vars
        states = []
        sum_reward = 0

        # 0. Step
        state, reward, done, info = self.env.step(action)

        # 1. bis n. Step
        for i in range(self.skip):

            # Wenn Env nicht abgeschlossen wurde
            if not done:
                # .. mach den Step
                state, reward, done, info = self.env.step(action)
                # .. bilde die Summe des Rewards
                sum_reward += reward
                # .. füg den State hinzu
                states.append(state)

            # Wenn Env abgeschlossen wurde
            else:
                # .. fülle mit den letzten State auf
                states.append(state)

        # Aus den gesammelten States ein Array bauen
        states = np.concatenate(states, 0)[None, :, :, :]

        # Genauigkeit reduzieren
        states = states.astype(np.float32)
        
        return states, reward, done, info

    def reset(self):
        # Weiterleiten
        state = self.env.reset()

        # State wiederholen und zu Array umformen
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]

        # Genauigkeit reduzieren
        states = states.astype(np.float32)

        return states


def preprocess_frame(frame):
    """Vereinfacht das übergebe frame"""
    if frame is not None:
        #cv2.imwrite("pre-process.jpg", frame) 

        # Frame zu Schwarz-Weiß (255 - 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #cv2.imwrite("post-rgb2gray.jpg", frame) 

        # Verkleinern 
        frame = cv2.resize(frame, (84, 84))
        #cv2.imwrite("post-resize.jpg", frame) 

        # Schwarz-Weiß zu Binary (1 - 0)
        frame = frame[None, :, :] / 255. 

        return frame

    # Ansonsten
    else:
        # Schwarzes Bild
        return np.zeros((1, 84, 84))


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
    env = SkipFrameWrapper(env, args.skip_frames)

    # Rückgabe
    num_states = env.observation_space.shape[0]
    num_actions = len(action_set)

    return env, num_states, num_actions

def make_testing_enviorment(args):
    """Erzeugt ein Testing-Enviorment"""

    # todo: Monitor

    return make_training_enviorment(args)
