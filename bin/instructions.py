# Generel
import os, shutil, time, glob
from datetime import datetime

# Torch
import torch as T
import torch.nn.functional as F
from torch.multiprocessing import Process

# Summary
from torchsummaryX import summary

# Hilfsklassen & -Funktionen
from bin.enviorment import make_training_enviorment, make_testing_enviorment
from bin.model import ActorCriticModel
from bin.optimizer import Adam
from bin.worker import dispatch_training, dispatch_testing
from bin.printers import *

os.environ['OMP_NUM_THREADS'] = '1' # Anzahl an Parrallelen Threads


def start_training(args):
    """Startet das Training, dazu werden unteranderem mehrere Trainings-Prozesse gestartet"""

    try:
        print(">> Fortschritt:\n\nInitialisiere Training...\nInitialisiere Torch-Multiprocessing...\nTorch-Seed: {}".format(args.torch_seed))

        # Bereite Torch-Multiprocessing vor
        T.manual_seed(args.torch_seed)
        mp = T.multiprocessing.get_context("spawn")

        # Initialisiere Trainings Enviorment
        env, num_states, num_actions = make_training_enviorment(args)

        print("Initialisiere Model...\nAnzahl an States (Input): {}\nAnzahl an Actions (Output): {}".format(num_states, num_actions))
        # Globales Model
        global_model = ActorCriticModel(num_states, num_actions)
        trained_episodes = 0

        # Wenn das Model geladen werden soll
        if args.model_load_latest:
            print("Suche Model-File Gewichte und Biase in \'{}\'...".format(args.modeldir))
            model_file, trained_episodes = get_corresponding_model_file(args)

            # Wenn File gefunden wurde
            if model_file is not None:
                print("Lade Model-File {}...".format(model_file))
                global_model.load_state_dict(T.load(model_file))

            # Wenn kein File gefunden wurde
            else:
                trained_episodes = 0
                print("Kein Model-File gefunden...".format(model_file))


        print("Bereite TensorboardX-Writer vor...")

        # SummaryWriter
        summarywriter_path = "{}/{}_w{}-s{}-v{}_{}".format(args.logdir, args.model_save_name, args.world, args.stage, args.rversion, datetime.now().strftime("%d-%m-%y_%H-%M-%S"))
        if not os.path.isdir(summarywriter_path):
            os.mkdir(summarywriter_path)

        print("Tensorboard kann mit dem CL-Kommando \'tensorboard --logdir={}\' gestartet werden..".format(args.logdir))

        # GPU-Support
        if args.cuda: 
            print("Aktiviere Cuda-GPU-Unterstüzung...")
            global_model.cuda() 

        # Gloables Model mit allen Teilen
        global_model.share_memory()

        print("Model Initallisierung abgeschlossen...\nInitialisiere Optimizer...")

        # Optimizer
        optimizer = Adam(global_model.parameters(), args)

        print("Optimizer Initallisierung abgeschlossen...\nInitialisiere Thread's...")

        # Threads
        threads = []

        if args.num_parallel_trainings_threads > 0:
            # initaler Trainings-Thread (wird abgespeichert)
            trainings_thread = mp.Process(target = dispatch_training, args = (0, args, global_model, optimizer, True, trained_episodes, summarywriter_path))
            threads.append(trainings_thread)

        # weitere Trainings-Threads
        for idx in range(1, args.num_parallel_trainings_threads):
            trainings_thread = mp.Process(target = dispatch_training, args = (idx, args, global_model, optimizer, False, trained_episodes, summarywriter_path))
            threads.append(trainings_thread)

        # Testing-Threads
        trainings_threads_count = args.num_parallel_trainings_threads
        for idx in range(args.num_parallel_testing_threads):
            test_thread = mp.Process(target = dispatch_testing, args = ((idx + trainings_threads_count), args, global_model, summarywriter_path))
            threads.append(test_thread)

        print("{} Thread's initialisiert ({} Worker und {} Tester)...\nStarten alle Threads in Kürze...\n".format(len(threads), args.num_parallel_trainings_threads, args.num_parallel_testing_threads))
        printStars("\n")

        print(">>> Neuronales-Netz:\n")

        # Beispiel-Input
        x = T.zeros((1, num_states, 64, 50))
        hx = T.zeros((1, 512))
        cx = T.zeros((1, 512))

        if args.cuda:
            x = x.cuda()
            hx = hx.cuda()
            cx = cx.cuda()

        # Infos von Torch
        print(global_model)
        print("\n")
        # Infos von TorchSummaryX
        summary(global_model, x, hx)

        print("\n")
        printStars("\n")
        print(">> Training:\n")

        # Starten aller Threads
        for t in threads:
            t.start()
            time.sleep(.25)

        # Synchronisieren
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        return


def start_testing(args):
    """Startet das Testing"""

    try:
        print(">> Fortschritt:\n\nInitialisiere Testing...\nInitialisiere Torch-Multiprocessing...\nTorch-Seed: {}".format(args.torch_seed))

        # Bereite Torch-Multiprocessing vor
        T.manual_seed(args.torch_seed)

        cuda = args.cuda

        model_files, episodes = get_all_corresponding_model_files(args)

        if model_files is None or episodes is None:
            print("\nKeine Model-Files für gegebene Argumente gefunden.\nBrechen Training ab...\n")
            return

        # Const
        modeldir = args.modeldir
        recordsdir = args.recordsdir
        model_save_name = args.model_save_name
        world = args.world
        stage = args.stage
        rversion = args.rversion
        num_parallel_trainings_threads = args.num_parallel_trainings_threads
        now_str = datetime.now().strftime("%d-%m-%y")
        counter = 1

        printStars("\n")
        print(">> Aufnahmen:")

        for model_file, episode in zip(model_files, episodes):

            print("\nBeginne Aufnahme {} / {}    --    World {}, Stage {}, Version {}, Episode {}    --    Model-Datei {}\n" \
                .format(counter, len(model_files), world, stage, rversion, episode, model_file))

            # Aufnahme-Enviorment initialisieren
            env, num_states, num_actions = make_testing_enviorment(args, episode)

            # Model laden
            local_model = ActorCriticModel(num_states, num_actions)
            local_model.load_state_dict(T.load(model_file))

            # Gpu-Support
            if cuda: local_model.cuda()

            # Evaluierungs-Flag
            local_model.eval()

            # Loop-Var
            local_state = T.from_numpy( env.reset() )
            local_done = True

            # Aufnahme-Loop
            while True:
                # Wenn Env abgeschlossen ist...
                if local_done: 
                    # Neue Tensor erzeugen
                    hx = T.zeros((1, 512), dtype = T.float)
                    #cx = T.zeros((1, 512), dtype = T.float) # LSTM-Version

                    # Enviorment zurücksetzen
                    env.reset()

                else:
                    # Tensor wiederverwenden
                    hx = hx.detach()
                    #cx = cx.detach() # LSTM-Version

                # GPU-Support
                if cuda:
                    hx = hx.cuda()
                    #cx = cx.cuda() # LSTM-Version
                    local_state = local_state.cuda()

                # Model
                #action_logit_probability, action_judgement, hx, cx = local_model(local_state, hx, cx) # LSTM-Version
                action_logit_probability, action_judgement, hx = local_model(local_state, hx) # GRU-Version

                # Policy
                policy = F.softmax(action_logit_probability, dim = 1)

                # Action wählen
                #action = int(T.argmax(policy).item()
                action = T.argmax(policy).item()

                # Ausführen
                local_state, local_reward, local_done, local_info = env.step(action)
                local_state = T.from_numpy(local_state)
                env.render()

                # Wenn der Run abgeschlossen ist
                if local_done:
                    try:
                        # Aufnahme stoppen
                        env.close()
                    except ValueError:
                        print("")
                    break

            # Warte kurz
            time.sleep(1)

            # Verschiebe die Aufnahme & Lösche den ursprünglichen Ordner
            for mp4 in glob.glob("{}/{}_world{}_stage{}_ver{}/ep{}/*1.mp4".format(recordsdir, model_save_name, world, stage, rversion, episode)):
                shutil.move(mp4, "{}/{}_world{}_stage{}_ver{}/ep_{}_x_{}__{}.mp4".format(recordsdir, model_save_name, world, stage, rversion, episode, num_parallel_trainings_threads, now_str))
                shutil.rmtree("{}/{}_world{}_stage{}_ver{}/ep{}/".format(recordsdir, model_save_name, world, stage, rversion, episode))

            print("\n.. abgeschlossen\n")
            counter += 1

    except KeyboardInterrupt:
        return


def get_corresponding_model_file(args):
    """Gibt das aktuellste (wenn nicht näher definiert) Model-File zurück für gegebene Argumente"""

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

    if args.load_model_from_prev_training:
        if stage == 1:
            world = world - 1
            stage = 4
        else:
            stage = stage - 1

    rversion = args.rversion
    num_parallel_trainings_threads = args.num_parallel_trainings_threads

    if not os.path.isdir("{}/{}_world{}_stage{}_ver{}".format(modeldir, model_save_name, world, stage, rversion)):
        return None, None

    # Initiale Filter World, Stage, RVersion
    matched_files = glob.glob("{}/{}_world{}_stage{}_ver{}/*.pt".format(modeldir, model_save_name, world, stage, rversion))

    # Überpüft ob passende Model-Files _nicht_ vorhanden waren
    if len(matched_files) == 0:
        return None, None

    # Wenn es nur 1 passenden gibt
    elif len(matched_files) == 1:
        file = matched_files[0]
        idx_0 = file.find('ep') + 2
        idx_1 = file.find('_x_')
        episode = int(file[idx_0:idx_1])

        return file, episode

    # Ansonsten finde das letzte
    else:
        try:
            biggest_episode = -1
            biggest_file = ""

            for file in matched_files:
                # Indicies
                idx_0 = file.find('ep') + 2
                idx_1 = file.find('_x_')
                #idx_2 = idx_1 + 3
                #idx_3 = file.find('.pt')

                episode = int(file[idx_0:idx_1])
                #thread = int(file[idx_2:idx_3])

                # mit <= damit das letzte in der 'Liste' genommen wird
                if biggest_episode <= episode:
                    biggest_episode = episode
                    biggest_file = file

            # Nochmaliges überprüfen ob files gefunden wurden
            if not biggest_file == "" and not biggest_episode == -1:
                return biggest_file, biggest_episode

            # Keines gefunden obwohl eins erwartet wurde
            else:
                print("Aktuellestes Model-File konnte nicht festgestellt werden.\n")
                return None, None

        except:
            print("Kritischer Parsing-Exception beim finden des Model-Files.\n"+\
                "Bitte stellen Sie sicher das alle im \'{}\' liegenden Model-Files folgendes Schema befolgen: \'{}_world<ZAHL>_stage<ZAHL>_ver<ZAHL>/ep<ZAHL>_x_<ZAHL>.pt\'\n"+\
                "-> Fahre ohne laden eines Models fort.\n".format(modeldir, model_save_name))
            return None, None


def get_all_corresponding_model_files(args):
    """Gibt alle passenden Model-Files zurück, als auch in einer seperaten Liste die episoden Anzahl des Models"""
    
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

    # Überprüfen ob der Ordner exisitert
    if not os.path.isdir("{}/{}_world{}_stage{}_ver{}".format(modeldir, model_save_name, world, stage, rversion)):
        return None, None

    # Initiale Filter World, Stage, RVersion
    matched_files = glob.glob("{}/{}_world{}_stage{}_ver{}/*.pt".format(modeldir, model_save_name, world, stage, rversion))

    # Überpüft ob passende Model-Files _nicht_ vorhanden waren
    if len(matched_files) == 0:
        return None, None

    # Ansonsten die Episoden extrahieren & als Liste zurückgegeben
    else:
        list_files = []
        list_episodes = []

        for file in matched_files:
            idx_0 = file.find('ep') + 2
            idx_1 = file.find('_x_')
            episode = int(file[idx_0:idx_1])

            list_files.append(file)
            list_episodes.append(episode)

        return list_files, list_episodes
