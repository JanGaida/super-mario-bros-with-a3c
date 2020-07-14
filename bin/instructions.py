""" © Jan Gaida, 2020 """

# Generel
import os, shutil, time, glob
from multiprocessing import Queue
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


""" instructions.py

Definiert die Abläufe der Programm-Modi (Training, Testing) und übernimmt dabei die fundamentalen Kontrollflussaufgaben


- start_training:
Erzeugt diverese Worker bzw. Tester und trainiert ein Netzwerk mit Hilfe des A3C-Algorithmus

- start_testing:
Testet trainierte Netzwerke in dem diese als MP4-File aufgenommen werden

- get_corresponding_model_file:
Sucht das letzte für die Argumente passende Model-File

- get_all_corresponding_model_files
Sucht alle passenden Model-Files für die gegeben Argumente

"""


os.environ['OMP_NUM_THREADS'] = '1'
memory_out_channels = 512
enviorment_out_height = 50
enviorment_out_width = 64


def start_training(args):
	"""Startet das Training, dazu werden unteranderem mehrere Trainings-Prozesse gestartet"""
	global memory_out_channels
	global enviorment_out_width
	global enviorment_out_height

	print(">> Starte Training, Fortschritt:\n")

	try:
		# Bereite Torch-Multiprocessing vor
		mp_context = "spawn"
		print("Initialisiere Torch-Multiprocessing ...\nSeed: {}\nContext: \'{}\'"
			.format(args.torch_seed, mp_context))
		T.manual_seed(args.torch_seed)
		mp = T.multiprocessing.get_context(mp_context)
		print("... initialisiert\n")

		# Initialisiere Trainings Enviorment
		print("Initialisiere Trainings-Enviorment ...\nWorld: {}, Stage: {}, Render-Version: {}\nBuffered-Frames: {}"
			.format(args.world, args.stage, args.rversion, args.skip_frames))
		env, num_states, num_actions = make_training_enviorment(args)
		print("Action-Set: {}, Number of Actions: {}\nOutput-Shape: {}\n... initialisiert\n"
			.format(args.action_set, num_actions, env.observation_space.shape))

		# Globales Model
		print("Initialisiere Globales-Model...")
		global_model = ActorCriticModel(num_states, num_actions)
		print("... initialisiert")
		trained_episodes = 0

		# Wenn das Model geladen werden soll
		if args.model_load_latest:
			print("... laden der Gewichte & Biase\n... suche Model-File in \'{}\'"
				.format(args.modeldir))
			model_file, trained_episodes = get_corresponding_model_file(args)

			# Wenn File gefunden wurde
			if model_file is not None:
				global_model.load_state_dict(T.load(model_file))
				print("... Model-File \'{}\' mit {} trainierten Episoden gefunden\n... laden der Gewichte & Biase erfolgreich"
					.format(model_file, trained_episodes))

			# Wenn kein File gefunden wurde
			else:
				trained_episodes = 0
				print("... Kein passendes Model-File gefunden\n... laden der Gewichte & Biase abgebrochen")

		else:
			print("... laden der Gewichte & Biase übersprungen")

		# GPU-Support
		if args.cuda: 
			global_model.cuda()
			print("... Cuda-GPU-Unterstüzung aktiviert")

		# Gloables Model mit allen Teilen
		global_model.share_memory()
		print("... Globales-Model einsatzbereit\n")

		# Optimizer
		print("Initialisiere Adam-Optimizer ...\nLearning-Rate: {}".format(args.learning_rate))
		optimizer = Adam(global_model.parameters(), args)
		print("... initialisiert\n")

		# SummaryWriter
		print("Initialsiere TensorboardX-Writer...")
		summarywriter_path = "{}/{}_w{}-s{}-v{}_{}".format(args.logdir, args.model_save_name, args.world, args.stage, args.rversion, datetime.now().strftime("%d-%m-%y_%H-%M-%S"))
		if not os.path.isdir(summarywriter_path):
			os.mkdir(summarywriter_path)
		print("Pfad: \'{}\'\nKommando zum starten des Tensorboard-Services: \'tensorboard --logdir={}\'\n... initialisiert\n"
			.format(summarywriter_path, args.logdir))


		# Threads
		print("Initialsiere Threads...\nAnzahl Worker: {}\nAnzahl Live-Tester: {}"
			.format(args.num_parallel_trainings_threads, args.num_parallel_testing_threads))

		worker_done_states = T.tensor([0])
		for it in worker_done_states:
			it.share_memory_()

		threads = []
		if args.num_parallel_trainings_threads > 0:
			# initaler Trainings-Thread (wird abgespeichert)
			trainings_thread = mp.Process(target = dispatch_training, args = (0, args, global_model, optimizer, True, trained_episodes, summarywriter_path, worker_done_states))
			threads.append(trainings_thread)

		# weitere Trainings-Threads
		for idx in range(1, args.num_parallel_trainings_threads):
			trainings_thread = mp.Process(target = dispatch_training, args = (idx, args, global_model, optimizer, False, trained_episodes, summarywriter_path, worker_done_states))
			threads.append(trainings_thread)

		# Testing-Threads
		trainings_threads_count = args.num_parallel_trainings_threads
		for idx in range(args.num_parallel_testing_threads):
			test_thread = mp.Process(target = dispatch_testing, args = ((idx + trainings_threads_count), args, global_model, summarywriter_path, worker_done_states))
			threads.append(test_thread)
		print("... insgesamt {} Threads initialisiert\n"
			.format(len(threads)))


		printStars("\n")


		print(">> Netzwerk Architektur:\n")

		# Beispiel-Input
		x = T.zeros((1, num_states, enviorment_out_height, enviorment_out_width))
		hx = T.zeros((1, memory_out_channels))
		cx = T.zeros((1, memory_out_channels))

		if args.cuda:
			x = x.cuda()
			hx = hx.cuda()
			cx = cx.cuda()

		# Infos von Torch
		print(global_model, "\n")

		# Infos von TorchSummaryX
		summary(global_model, x, hx, cx) # LSTM-Version
		#summary(global_model, x, hx) # GRU-Verison

		print("\n")
		printStars("\n")
		print(">> Training:\n")

		# Starten aller Threads
		print("Starte alle Threads ...")
		for idx, t in enumerate(threads):
			t.start()
			print("... #{} Thread gestartet"
				.format(idx))
			time.sleep(.1)
		print("... fertig\n")

		# Synchronisieren
		for t in threads:
			t.join()

	except KeyboardInterrupt:
		return


def start_testing(args):
	"""Startet das Testing"""
	global memory_out_channels

	print(">> Starte Testing, Fortschritt:\n")

	try:
		# Seed
		print("Initialisiere Model ...\nSeed: {}".format(args.torch_seed))
		T.manual_seed(args.torch_seed)

		# Cuda
		cuda = args.cuda
		if cuda:
			print("CUDA: Aktiv")
		else: 
			print("CUDA: Inaktiv")

		# Model-Files
		print("... suche nach passenden Model-Files")
		model_files, episodes = get_all_corresponding_model_files(args)

		if model_files is None:
			print("... keine passendes Model-File gefunden\n... abbruch")
			return
		if episodes is None:
			print("... keine Episoden-Anzahl geparst\n... abbruch")
			return
		print("... {} Model-Files gefunden\n"
			.format(len(model_files)))

		print("Bereit Aufnahmen vor ...")        
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
		print("World: {}, Stage: {}, Render-Version: {}\nTraing-Threads: {}\nModel-Dir: \'{}\'\nAufnahmen-Dir: \'{}\'\n... abgeschlossen\n"
			.format(world, stage, rversion, num_parallel_trainings_threads, modeldir, recordsdir))

		printStars("\n")
		print(">> Aufnahmen:")

		for model_file, episode in zip(model_files, episodes):

			print("\nBeginne Aufnahme {} / {} ...\nWorld: {}, Stage: {}, Version: {}, Episode: {}\nModel-Datei: \'{}\'" \
				.format(counter, len(model_files), world, stage, rversion, episode, model_file))

			# Aufnahme-Enviorment initialisieren 
			env, num_states, num_actions = make_testing_enviorment(args, episode)
			print("... Testing-Enviorment initialisiert")

			# Model laden
			print("... lade lokales Model")
			local_model = ActorCriticModel(num_states, num_actions)
			local_model.load_state_dict(T.load(model_file))
			# Gpu-Support
			if cuda: local_model.cuda()
			# Evaluierungs-Flag
			local_model.eval()
			print("... evaluiere")

			# Loop-Var
			local_state = T.from_numpy( env.reset() )
			local_done = True

			# Aufnahme-Loop
			while True:
				# Wenn Env abgeschlossen ist...

				# LSTM-Version
				if local_done: 
					# Neue Tensor erzeugen
					hx = T.zeros((1, memory_out_channels), dtype = T.float)
					cx = T.zeros((1, memory_out_channels), dtype = T.float)
					# Enviorment zurücksetzen
					env.reset()
				else:
					# Tensor wiederverwenden
					hx = hx.detach()
					cx = cx.detach()
				# GPU-Support
				if cuda:
					hx = hx.cuda()
					cx = cx.cuda() 
					local_state = local_state.cuda()
				# Model
				action_logit_probability, action_judgement, hx, cx = local_model(local_state, hx, cx)

				""" # GRU-Version
				if local_done: 
					# Neue Tensor erzeugen
					hx = T.zeros((1, memory_out_channels), dtype = T.float)
					# Enviorment zurücksetzen
					env.reset()
				else:
					# Tensor wiederverwenden
					hx = hx.detach()
				# GPU-Support
				if cuda:
					hx = hx.cuda()
					local_state = local_state.cuda()
				# Model
				action_logit_probability, action_judgement, hx = local_model(local_state, hx)
				"""
				
				# Policy
				policy = F.softmax(action_logit_probability, dim = 1)

				# Action wählen
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
			print("... fertig\n... formatiere Aufnahme")
			time.sleep(1)

			# Verschiebe die Aufnahme & Lösche den ursprünglichen Ordner
			for mp4 in glob.glob("{}/{}_world{}_stage{}_ver{}/ep{}/*1.mp4".format(recordsdir, model_save_name, world, stage, rversion, episode)):
				shutil.move(mp4, "{}/{}_world{}_stage{}_ver{}/ep_{}_x_{}__{}.mp4".format(recordsdir, model_save_name, world, stage, rversion, episode, num_parallel_trainings_threads, now_str))
				shutil.rmtree("{}/{}_world{}_stage{}_ver{}/ep{}/".format(recordsdir, model_save_name, world, stage, rversion, episode))

			print("... abgeschlossen\n")
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

				episode = int(file[idx_0:idx_1])

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

	# ansonsten das letzte passende ..
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
