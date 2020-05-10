# Generel
import argparse
import shutil
import os

# Hilfsklassen & -Funktionen
from bin.printers import printHeaders, printTrainingMode, printTestingMode
from bin.instructions import start_training, start_testing

def grab_arguments():
	"""Hilfsfunktion um Argumente zu parsen"""

	parser = argparse.ArgumentParser("Research Project: Deep-Q-Learning mit 'Super Mario Bros' und A3C.")
	
	# Modus
	parser.add_argument("-m", "--mode", type=str, default="training", 
		help="Wertebereich: \'training\', \'testing\'\nLegt fest ob ein Model trainiert oder getestet werden soll.")
	parser.add_argument("-v", "--verbose", type=bool, default=True, 
		help="Ob diverse Informationen ausgegeben werden sollen.")
	parser.add_argument("--verbose_every_episode", type=int, default=10, 
		help="Nach welchen Episoden diverse Informationen ausgegeben werden sollen.")


	# Ordner
	parser.add_argument("--logdir", type=str, default="logs", 
		help="Legt den Pfad fest in dem Logs abgelegt werden.")
	parser.add_argument("--rm_logdir", type=bool, default=False, 
		help="Legt fest ob der Logging-Pfad geleert werden soll wenn vorhanden.")
	parser.add_argument("--modeldir", type=str, default="models", 
		help="Legt den Pfad fest in dem Models gespeichert werden.")
	parser.add_argument("--recordsdir", type=str, default="rec", 
		help="Legt den Pfad fest in dem Aufnahmen gespeichert werden.")



	# Laufzeitumgebung
	parser.add_argument("--cuda", type=bool, default=True, 
		help="Legt fest ob die GPU (mit CUDA-Support) verwendet werden soll.")
	parser.add_argument("--num_parallel_trainings_threads", type=int, default=6, 
		help="Legt fest wie viele Threads verwendet werden sollen.")
	parser.add_argument("--num_parallel-testing_threads", type=int, default=1, 
		help="Legt fest wie viele Threads verwendet werden sollen.")


	# Torch
	parser.add_argument("--torch_seed", type=int, default=42, 
		help="Der initiale Torchseed.")
	parser.add_argument("--model_save_name", type=str, default="a3c_smb",
		help="Der Name des gespeichert Models (Anmerkung: wird um die Stage, World-, Version- sowie Step- und Thread-Informationen ergänzt).")
	parser.add_argument("--model_load_latest", type=bool, default=True,
		help="Ob ein Model beim Start geladen werden soll.")
	parser.add_argument("--model_load_file", type=str, default="",
		help="Der Name des Models welches geladen werden soll.")


	# Enviorment
	parser.add_argument("-w", "--world", type=int, default=1, 
		help="Legt die initiale Welt des Super-Mario-Bros-Gym's fest.")
	parser.add_argument("-s", "--stage", type=int, default=1, 
		help="Legt die initiale Stage des Super-Mario-Bros-Gym's fest.")
	parser.add_argument("-rv", "--rversion", type=int, default=0, 
		help="Legt die initiale Version des Super-Mario-Bros-Gym's fest.")
	parser.add_argument("-a", "--action_set", type=str, default="complex",
		help="Wertebereich: \'rightonly\', \'simple\', \'complex\'\nLegt die initiale Aktionen des Super-Mario-Bros-Gym's fest.")
	parser.add_argument("--episode_save_interval", type=int, default=500,
		help="Anzahl an Episoden nach den das globale Model gespeichert werden soll.")
	parser.add_argument("--max_local_steps", type=int, default=50,
		help="Die maximale Anzahl an lokalen Steps die ein Worker auszuführen hat.")
	parser.add_argument("--max_global_steps", type=int, default=5e5,
		help="Die maximale Anzahl an globalen Steps die ein Worker auszuführen hat.")
	parser.add_argument("--max_actions", type=int, default=100, 
		help="Maximale Wiederholung von Aktionen in der Testphase.")
	parser.add_argument("--skip_frames", type=int, default=5, 
		help="Die Anzahl an Frames die Übersprungen werden.")


	# Hyperparameter
	parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
		help="Learningrate-Faktor.")
	parser.add_argument('--discount_gamma', type=float, default=0.9,
		help='Discount- bzw. Gamma-Faktor.')
	parser.add_argument('--tau', type=float, default=1.0, 
		help='Parameter für GAE.')
	parser.add_argument('--beta', type=float, default=0.01, 
		help='Entropy Koeffizient.')

	return parser.parse_args()


def check_dir_structure(args):
	"""Überprüft ob die gewünschten Ordner vorhanden sind"""
	
	# logdir
	if args.rm_logdir and os.path.isdir(args.logdir):
		shutil.rmtree(args.logdir)
	if not os.path.isdir(args.logdir):
		os.makedirs(args.logdir)

	# modeldir
	if not os.path.isdir(args.modeldir):
		os.makedirs(args.modeldir)

	# recordsdir
	if not os.path.isdir(args.recordsdir):
		os.makedirs(args.recordsdir)

if __name__ == "__main__":
	"""Die Main-Funktion"""

	# alle benötigten Argumente
	args = grab_arguments()

	# Headers
	printHeaders(args)

	# Ordnerstruktur
	check_dir_structure(args)

	# In den Modus wechseln
	if args.mode == "training":
		# TRAINING
		printTrainingMode()
		start_training(args)

	elif args.mode == "testing":
		# TESTING
		printTestingMode()
		start_testing(args)

	else:
		print("Invalider Modus. Verfügbare Modi sind \'training\' oder \'testing\'.\n\nEXIT: OK")
