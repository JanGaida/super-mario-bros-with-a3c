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

"""
class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())
"""

def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class RewardWrapper(Wrapper):
    def __init__(self, env=None):
        super(RewardWrapper, self).__init__(env)
        self.observation_space = Box(low = 0, high = 255, shape = (1, 84, 84), dtype = np.float32)
        self.curr_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class SkipFrameWrapper(Wrapper):
    def __init__(self, env, skip=4):
        super(SkipFrameWrapper, self).__init__(env)
        self.observation_space = Box(low = 0, high = 255, shape = (4, 84, 84), dtype = np.float32)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
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

    # Überschreiben des Rewards
    env = RewardWrapper(env)

    # Überspringen von Frames
    env = SkipFrameWrapper(env)

    # Rückgabe
    num_states = env.observation_space.shape[0]
    num_actions = len(action_set)

    return env, num_states, num_actions

def make_testing_enviorment(args):
    """Erzeugt ein Testing-Enviorment"""
    return make_training_enviorment(args)
