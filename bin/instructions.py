# Generel
import os, shutil, time, glob

# Torch
import torch as T
import torch.nn.functional as F
from torch.multiprocessing import Process

# Hilfsklassen & -Funktionen
from bin.enviorment import make_training_enviorment, make_testing_enviorment
from bin.model import ActorCriticModel
from bin.optimizer import Adam
from bin.worker import dispatch_training, dispatch_testing
from bin.printers import *

# Anzahl an Parrallelen Threads
os.environ['OMP_NUM_THREADS'] = '1'

def start_training(args):
    """Startet das Training, dazu werden unteranderem mehrere Trainings-Prozesse gestartet"""

    print(">> Fortschritt:\n\nInitialisiere Training...\nInitialisiere Torch-Multiprocessing...\nTorch-Seed: {}".format(args.torch_seed))

    # Bereite Torch-Multiprocessing vor
    T.manual_seed(args.torch_seed)
    mp = T.multiprocessing.get_context("spawn")

    # Initialisiere Trainings Enviorment
    env, num_states, num_actions = make_training_enviorment(args)

    print("Initialisiere Model...\nAnzahl an States (Input): {}\nAnzahl an Actions (Output): {}".format(num_states, num_actions))
    # Globales Model
    global_model = ActorCriticModel(num_states, num_actions)

    # Wenn das Model geladen werden soll
    if args.model_load_latest:
        print("Suche Model-File Gewichte und Biase in \'{}\'...".format(args.modeldir))
        model_file = get_corresponding_model_file(args)

        # Wenn File gefunden wurde
        if model_file is not None:
            print("Lade Model-File {}...".format(model_file))
            global_model.load_state_dict(T.load(model_file))

        # Wenn kein File gefunden wurde
        else:
            print("Kein Model-File gefunden...".format(model_file))


    # GPU-Support
    if args.cuda: 
        print("Aktiviere Cuda-Unterstüzung...")
        global_model.cuda() 

    # Gloables Model mit allen Teilen
    global_model.share_memory()

    print("Model Initallisierung abgeschlossen...\nInitialisiere Optimizer...")

    # Optimizer
    optimizer = Adam(global_model.parameters(), args)

    print("Optimizer Initallisierung abgeschlossen...\nInitialisiere Thread's...")

    # Threads
    threads = []

    # initaler Trainings-Thread (wird abgespeichert)
    trainings_thread = mp.Process(target = dispatch_training, args = (0, args, global_model, optimizer, True))
    threads.append(trainings_thread)

    # weitere Trainings-Threads
    for idx in range(1, args.num_parallel_trainings_threads):
        trainings_thread = mp.Process(target = dispatch_training, args = (idx, args, global_model, optimizer, False))
        threads.append(trainings_thread)

    # Testing-Threads
    trainings_threads_count = args.num_parallel_trainings_threads
    for idx in range(args.num_parallel_testing_threads):
        test_thread = mp.Process(target = dispatch_testing, args = ((idx + trainings_threads_count), args, global_model))
        threads.append(test_thread)

    print("{} Thread's initialisiert ({} Worker und {} Tester)...\nStarte Trainings-Threads...\n".format(len(threads), args.num_parallel_trainings_threads, args.num_parallel_testing_threads))
    printStars("\n")
    print(">> Training:\n")

    # Starten aller Threads
    for t in threads:
        t.start()
        time.sleep(.25)

    try:
        # Synchronisieren
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        return

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
        local_model.load_state_dict(T.load("{}/{}_world{}_stage{}_ver{}__ep{}_x_{}.pt".format(modeldir, model_save_name, world, stage, rversion, target_episode, target_parralel_threads)))
        local_model.cuda()
    else:
        local_model.load_state_dict(T.load("{}/{}_world{}_stage{}_ver{}__ep{}_x_{}.pt".format(modeldir, model_save_name, world, stage, rversion, target_episode, target_parralel_threads), map_location = lambda storage, loc: storage))
    
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


def get_corresponding_model_file(args):
    """"""
    modeldir = args.modeldir
    model_load_file = args.model_load_file

    # Wenn ein spezifisches File vorgegeben wurde
    if not model_load_file == "":
        # Überprüfe ob es existiert
        if os.path.isfile("{}/{}".format(modeldir, model_load_file)):
            return model_load_file
        # ansonsten das letzte passende

    model_save_name = args.model_save_name
    world = args.world
    stage = args.stage
    rversion = args.rversion
    num_parallel_trainings_threads = args.num_parallel_trainings_threads

    # Initiale Filter World,Stage,RVersion
    matched_files = []
    for file in glob.glob("{}/*.pt".format(modeldir)):
        # Thread
        if file.startswith("{}_world{}_stage{}_ver{}".format(model_save_name, world, stage, rversion)):
            matched_files.append(file)

    # Überpüft ob passende Model-Files _nicht_ vorhanden waren
    if len(matched_files) == 0:
        return None

    # Wenn es nur 1 passenden gibt
    elif len(matched_files) == 1:
        return matched_files[0]

    # Ansonsten finde das letzte
    else:
        try:
            biggest_training = -1
            biggest_file = ""

            for file in matched_files:
                # Indicies
                idx_0 = file.find('__ep') + 4
                idx_1 = file.find('_x_')
                idx_2 = idx_1 + 3
                idx_3 = file.find('.pt')

                episode = int(file[idx_0:idx_1])
                thread = int(file[idx_2:idx_3])

                training = episode * thread

                # mit <= damit das letzte in der 'Liste' genommen wird
                if biggest_training <= training:
                    biggest_training = training
                    biggest_file = file

            # Nochmaliges überprüfen ob files gefunden wurden
            if not biggest_training == -1 and not biggest_file == "":
                return biggest_file

            # Keines gefunden obwohl eins erwartet wurde
            else:
                print("Aktuellestes Model-File konnte nicht festgestellt werden.\n")
                return None

        except:
            print("Kritischer Parsing-Exception beim finden des Model-Files.\n\
                Bitte stellen Sie sicher das alle im \'{}\' liegenden Model-Files folgendes Schema befolgen: {}_world<ZAHL>_stage<ZAHL>_ver<ZAHL>__ep<ZAHL>_x_<ZAHL>.pt\n\
                -> Fahre ohne laden eines Models fort.\n".format(modeldir, model_save_name))
            return None
