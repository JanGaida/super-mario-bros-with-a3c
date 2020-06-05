# Generel
import numpy as np
import cv2

# Gym
import gym
from gym import Wrapper
from gym.spaces import Box
from gym.wrappers import Monitor

# Super-Mario-Bros-Gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

np.seterr(over = 'ignore') # Ignoriere Numpy-Warnings -> https://github.com/RunzheYang/MORL/issues/5
gym.logger.set_level(40) # Gym-Logger-Level -> DEBUG = 10  INFO = 20  WARN = 30  ERROR = 40  DISABLED = 50


class PreprocessFrameWrapper(Wrapper):
    """Bearbeitet alle Frames des Enviorments"""

    def __init__(self, env):
        super(PreprocessFrameWrapper, self).__init__(env)
        """Init"""

        # Write-Img-Parameter
        #self.wrote_image_counter = 0
        #self.write_every_x_image = 1000
        #self.max_wrote_image_counter = 10001


    def step(self, action):
        """leite den Step weiter"""

        state, reward, done, info = self.env.step(action) # den Step auffangen
        return self.preprocess_frame(state), reward, done, info # den Frame verarbeiten


    def reset(self):
        """leitet den Reset-Call weiter"""

        return self.preprocess_frame(self.env.reset()) # Leite den call weiter und verarbeite den Frame


    def preprocess_frame(self, frame):
        """Vereinfacht das übergebe Frame"""

        # frame.shape == (240, 256, 3) __Auflösung: 256 / 240 == 16 / 15
        if frame is not None:

            #if (not self.wrote_image_counter == self.max_wrote_image_counter) and self.wrote_image_counter % self.write_every_x_image == 0:
            #    cv2.imwrite("frame-pre-processing-{}.jpg".format(self.wrote_image_counter), frame)

            # Zuschneiden __ frame.shape == (200, 256, 3) __ Auflösung: 256/200 == 32/25
            frame = frame[15:215,:]

            #if (not self.wrote_image_counter == self.max_wrote_image_counter) and self.wrote_image_counter % self.write_every_x_image == 0:
            #    cv2.imwrite("img/frame-cut-{}.jpg".format(self.wrote_image_counter), frame)

            # Frame zu Schwarz-Weiß (255 - 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #if (not self.wrote_image_counter == self.max_wrote_image_counter) and self.wrote_image_counter % self.write_every_x_image == 0:
            #    cv2.imwrite("img/frame-black-n-white-{}.jpg".format(self.wrote_image_counter), frame)

            # Treshold anwenden
            frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)

            #if (not self.wrote_image_counter == self.max_wrote_image_counter) and self.wrote_image_counter % self.write_every_x_image == 0:
            #    cv2.imwrite("img/frame-w-gau-tresh-{}.jpg".format(self.wrote_image_counter), frame)

            # Verkleinern __ frame.shape == (50, 64) __ Auflösung: 64/50 == 32/25
            frame = cv2.resize(frame, (64, 50))

            #if (not self.wrote_image_counter == self.max_wrote_image_counter) and self.wrote_image_counter % self.write_every_x_image == 0:
            #    cv2.imwrite("img/frame-resized-{}.jpg".format(self.wrote_image_counter), frame) 

            # Schwarz-Weiß zu Binary (1 - 0) & Channel hinzufügen __ frame.shape == (1, 50, 64)
            frame = frame[None, :, :] / 255. 

            #if not self.wrote_image_counter == self.max_wrote_image_counter:
            #    self.wrote_image_counter += 1

            return frame

        # Ansonsten
        else:
            # Leeres Bild
            return np.zeros((1, 50, 64))


class RewardWrapper(Wrapper):
    """Überschreibt die Reward-Funktion"""

    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)
        """Init"""

        self.x_0 = 40 # Mario's initiale X-Position
        self.score_0 = 0 # Mario's initialer Score
        self.clock_0 = 400 # Mario's initiale Zeit, Anmerkung: In Bowser-Lvln 300
        self.life_0 = 3 # Mario's initiale Leben
        #self.coins_0 = 0  # Mario's initialer Coins
        #self.status_0 = 0 # Mario's initaler Status (== small)


    def step(self, action):
        """Leitet den Step call weiter und übschreibt den Reward"""

        # Info-Dict ~> https://github.com/Kautenja/gym-super-mario-bros#info-dictionary
        # Org. Reward ~> https://github.com/Kautenja/gym-super-mario-bros#reward-function

        # leite den Step weiter
        state, reward, done, info = self.env.step(action)

        # VARs grabn
        x_1 = info['x_pos']
        score_1 = info['score']
        clock_1 = info['time']
        life_1 = info['life']

        reward =  ( max( x_1 - self.x_0, -5 ) ) \
                + ( max( score_1 - self.score_0, 0 ) / 400. ) \
                + ( clock_1 - self.clock_0 ) / 10. \
                + ( 0. if not done else  50. if info['flag_get'] else -50.) \
                + ( -50. if not life_1 == self.life_0 else 0. )

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
        self.life_0 = life_1

        """ Gut für W1S2        
        reward =  ( max( x_1 - self.x_0, -1 ) / 2. ) \
                + ( max(clock_1 - self.clock_0, -1) ) \
                + ( max( score_1 - self.score_0, 0 ) / 400. ) \
                + ( 0. if not done else  50. if info['flag_get'] else -50.) \
                + ( -5 if not life_1 == self.life_0 else 0 )
        """

        """ Sehr gut für W1S1
        reward =  ( max( x_1 - self.x_0, 0 ) ) \
                + ( max( score_1 - self.score_0, 0 ) / 400. ) \
                + ( clock_1 - self.clock_0 ) / 10. \
                + ( 0. if not done else  50. if info['flag_get'] else -50.)
        """

        """ Semi-Gut
        reward = \
                  ( max( x_1 - self.x_0, 0 ) ) / 10  \
                + ( max( score_1 - self.score_0, 0 ) / 1000. ) \
                + ( clock_1 - self.clock_0 ) * 2 \
                + ( 10 if done and info['flag_get'] else 0 ) \
                # + ( -5 if not life_1 == self.life_0 else 0 )
        """

        # Fertig
        return state, reward / 10. , done, info        


    def reset(self):
        """Leitet den Reset-Call weiter"""

        # Letzten Score ebenfalls zurücksetzten
        self.score_0 = 0
        #self.coin_0 = 0

        # Weiterleiten
        return self.env.reset()


    def status_to_int(self, status):
        """Hilfsfunktion um den Status von Mario in einen vergleichbaren Integer zu wandeln"""

        if not status == "small":
            return 1
        else:
            return 0


class FrameBufferWrapper(Wrapper):
    """Puffert einige Frames in einem Numpy-Array, überschreibt den Observation_Space dementsprechend"""

    def __init__(self, env, skip):
        super(FrameBufferWrapper, self).__init__(env)
        """Init"""

        # Merk wie viel Frames übersprungen werden sollen
        self.skip = (skip - 1)

        # Überschreib den Observation_Space
        self.observation_space = Box(low = 0, high = 255, shape = (4, 50, 64), dtype = np.float32)


    def step(self, action):
        """Leite den Step-Call weiter und puffert die Frames"""

        # zusammelnde VARs
        states = []
        sum_reward = 0

        # 0. Step
        state, reward, done, info = self.env.step(action)

        # 1. bis n. Step
        for _ in range(self.skip):

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

        # F-Genauigkeit reduzieren
        states = states.astype(np.float32)

        return states, reward, done, info


    def reset(self):
        """Leitet den Reset-Call weiter"""

        # Weiterleiten
        state = self.env.reset()
        # State wiederholen und zu Array umformen
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        # Genauigkeit reduzieren
        states = states.astype(np.float32)

        return states


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
