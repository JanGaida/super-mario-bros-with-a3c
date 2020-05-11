# Generel
import numpy as np
import cv2
import subprocess as sp

# Gym
import gym
from gym import Wrapper
from gym.spaces import Box
from gym.wrappers import Monitor

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
        self.x_0 = 40 # Mario's initiale X-Position
        self.score_0 = 0 # Mario's initialer Score
        #self.coins_0 = 0  # Mario's initialer Coins
        self.clock_0 = 400 # Mario's initiale Zeit
        self.life_0 = 3 # Mario's initiale Leben
        #self.status_0 = 0 # Mario's initaler Status (== small)

        # prev imp
        #self.score_0 = 0
        #self.coin_0 = 0

    def step(self, action):
        # Info-Dict ~> https://github.com/Kautenja/gym-super-mario-bros#info-dictionary
        # Org. Reward ~> https://github.com/Kautenja/gym-super-mario-bros#reward-function

        # leite den Step weiter
        state, reward, done, info = self.env.step(action)

        # hier berrechnet Reward liegt zwischen -15 und 15 
        # reward = (x_pos_1 - x_pos_0) + (clock_0 - clock_1) + (alive ? 0 ansonsten -15)

        reward = 0

        # X-POSITION
        x_1 = info['x_pos']
        score_1 = info['score']
        clock_1 = info['time']
        #life_1 = info['life']

        # Formula
        reward += max(x_1 - self.x_0, -0.01) + (max(score_1 - self.score_0, 0) / 10) + ((clock_1 - self.clock_0) * 2)

        # + Leben-Bestrafung
        #if not life_1 == self.life_0:
        #    reward += -5

        # + Ziel erreicht
        if done:
            # Das Env ist abgeschossen:
            if info['flag_get']:
                # Mario hat die Flagge erreicht
                reward += 200
            #else:
            #    # Mario hat nicht die Flagge erreicht
            #    reward += -50

        self.x_0 = x_1
        self.score_0 = score_1
        #self.coins_0 = coins_1
        self.clock_0 = clock_1
        #self.life_0 = life_1
        #self.status_0 = status_1

        #coins_1 = info['coins'] # Annmerkung: 1 Coin == 200 Score
        #delta_coins = coins_1 - self.coins_0

        #status_1 = self.status_to_int(info['status'])
        #delta_status = status_1 - self.status_0

        #f delta_coins > 0:
        #    # Mario hat Coins gesammelt
        #    reward += 0.35 # + 0.6 durch Score

        #if delta_status > 0:
        #    # Mario ist gewachsen
        #    reward += 0.2
        #else:
        #    # Mario ist geschrumpft
        #    reward += -0.2

        #if delta_life == -1:
        #    # Mario hat ein Leben verloren
        #    reward += -15


        # Letzte Werte merken


        # Fertig
        return state, reward, done, info

        """
        Alte Reward-Implementation ... ~~~ Funktioniert, könnte aber schneller gehen -> Problem im lvl 2

        my_reward = 0
        # zusätzlicher Reward mindestens 0
        # my_reward = (score_1 - self.score_0) / 50. + (coin_1 - self.coin_0) + (complete ? 45 ansonsten -45)
        # ~> reward insgesamt: (reward + my_reward) / 10.

        # Score:
        score_1 = info["score"]
        my_reward += (score_1 - self.score_0) / 50.

        # Coins
        coin_1 = info["coins"]
        my_reward += (coin_1 - self.coin_0) / 10.

        self.score_0 = score_1
        self.coin_0 = coin_1

        my_reward = max(0, my_reward)

        # Wenn das Enviorment abgeschlossen ist
        if done:
            # Und das Ziel erreicht wurde
            if info["flag_get"]:
                my_reward += 45.

            # Und das Ziel _nicht_ erreicht wurde
            else:
                my_reward += -45.

        reward += my_reward

        return state, reward / 10. , done, info
        """

    def reset(self):
        # Letzten Score ebenfalls zurücksetzten
        self.score_0 = 0
        self.coin_0 = 0

        # Weiterleiten
        return self.env.reset()

    def status_to_int(self, status):
        if not status == "small":
            return 1
        else:
            return 0


class BufferSkipFrameWrapper(Wrapper):

    def __init__(self, env, skip):
        super(BufferSkipFrameWrapper, self).__init__(env)
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
    env = BufferSkipFrameWrapper(env, args.skip_frames)

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
    env = BufferSkipFrameWrapper(env, args.skip_frames)

    # Rückgabe
    num_states = env.observation_space.shape[0]
    num_actions = len(action_set)

    return env, num_states, num_actions
