# Generel
import os
import argparse
import shutil
import time

# Torch
import torch as T
import torch.nn.functional as F
from torch.multiprocessing import Process

# Hilfsklassen & -Funktionen
from bin.enviorment import make_training_enviorment, make_testing_enviorment
from bin.model import ActorCriticModel
from bin.optimizer import Adam
from bin.worker import dispatch_training, dispatch_testing

# Anzahl an Parrallelen Threads
os.environ['OMP_NUM_THREADS'] = '1'

def start_training(args):
    """Startet das Training, dazu werden unteranderem mehrere Trainings-Prozesse gestartet"""

    # Bereite Torch-Multiprocessing vor
    T.manual_seed(args.torch_seed)
    mp = T.multiprocessing.get_context("spawn")

    # Initialisiere Trainings Enviorment
    env, num_states, num_actions = make_training_enviorment(args)

    # Globales Model
    global_model = ActorCriticModel(num_states, num_actions)
    if args.cuda: global_model.cuda() # GPU-Support
    global_model.share_memory()

# todo: Prev-Gewichte ? 

    # Optimizer
    optimizer = Adam(global_model.parameters(), args)

    # Threads
    threads = []

    # initaler Trainings-Thread (wird abgespeichert)
    trainings_thread = mp.Process(target = dispatch_training, args = (0, args, global_model, optimizer, True))
    threads.append(trainings_thread)

    # weitere Trainings-Threads
    for idx in range(args.num_parallel_trainings_threads):
        trainings_thread = mp.Process(target = dispatch_training, args = (idx, args, global_model, optimizer, False))
        threads.append(trainings_thread)

    # Testing-Threads
    trainings_threads_count = args.num_parallel_trainings_threads
    for idx in range(args.num_parallel_testing_threads):
        test_thread = mp.Process(target = dispatch_testing, args = ((idx + trainings_threads_count), args, global_model))
        threads.append(test_thread)

    # Starten aller Threads
    for t in threads:
        t.start()
        time.sleep(.25)

    # Synchronisieren
    for t in threads:
        t.join()

def start_testing(args):
    """Startet das Testing"""

    # Bereite Torch-Multiprocessing vor
    T.manual_seed(args.torch_seed)

    # Enviorment initialisieren
    env, num_states, num_actions = make_testing_enviorment(args)

    # Cuda-Unterstüzung
    cuda = args.cuda

    # Model initialisieren
    local_model = ActorCriticModel(num_states, num_actions)

    # Const
    modeldir = args.modeldir
    model_save_name = args.model_save_name
    world = args.world
    stage = args.stage
    rversion = args.rversion

    # Gewichte laden
    target_episode, target_parralel_threads = "5000", "6"
    if cuda:
        local_model.load_state_dict(T.load("{}/{}_world{}_stage{}_ver{}__ep{}_x_{}".format(modeldir, model_save_name, world, stage, rversion, target_episode, target_parralel_threads)))
        local_model.cuda()
    else:
        local_model.load_state_dict(T.load("{}/{}_world{}_stage{}_ver{}__ep{}_x_{}".format(modeldir, model_save_name, world, stage, rversion, target_episode, target_parralel_threads), map_location = lambda storage, loc: storage))
    
    # Evaluations-Flag
    local_model.eval()

    # Loop-Var
    local_state = T.from_numpy( env.reset() )
    local_done = True

    # Testing-Loop
    while True:
        # Wenn Env abgeschlossen ist...
        if local_done: 
            # Neue Tensor erzeugen
            h_0 = T.zeros((1, 512), dtype = T.float)
            c_0 = T.zeros((1, 512), dtype = T.float)

            # Enviorment zurücksetzen
            env.reset()

        else:
            # Tensor wiederverwenden
            h_0 = h_0.detach()
            c_0 = c_0.detach()

        # GPU-Support
        if cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            local_state = local_state.cuda()

        # Model
        action_logit_probability, action_judgement, h_0, c_0 = local_model(local_state, h_0, c_0)

        # Policy
        policy = F.softmax(action_logit_probability, dim = 1)

        # Action wählen
        #action = int(T.argmax(policy).item()
        action = T.argmax(policy).item()

        # Ausführen
        local_state, local_reward, local_done, local_info = env.step(action)
        local_state = T.from_numpy(local_state)
        env.render()

        if local_info["flag_get"]:
            print("World {} stage {} completed".format(world, stage))
            break
