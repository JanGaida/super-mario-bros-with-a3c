def printHeaders(args):
    """Gibt den Header des Skriptes aus."""
    printStars()
    printProjektInfos()
    printStars()
    print("                               ______                              _          ______          _           _         \n\
                               | ___ \                            | |         | ___ \        (_)         | |  _     \n\
                               | |_/ /___  ___  ___  __ _ _ __ ___| |__ ______| |_/ / __ ___  _  ___  ___| |_(_)    \n\
                               |    // _ \/ __|/ _ \/ _` | '__/ __| '_ \______|  __/ '__/ _ \| |/ _ \/ __| __|      \n\
                               | |\ \  __/\__ \  __/ (_| | | | (__| | | |     | |  | | | (_) | |  __/ (__| |_ _     \n\
                               \_| \_\___||___/\___|\__,_|_|  \___|_| |_|     \_|  |_|  \___/| |\___|\___|\__(_)    \n\
                                                                                            _/ |                    \n\
                                                                                           |__/                     \n\
                                _____                        ___  ___           _        ______                     \n\
                               /  ___|                       |  \/  |          (_)       | ___ \                    \n\
                               \ `--. _   _ _ __   ___ _ __  | .  . | __ _ _ __ _  ___   | |_/ /_ __ ___  ___       \n\
                                `--. \ | | | '_ \ / _ \ '__| | |\/| |/ _` | '__| |/ _ \  | ___ \ '__/ _ \/ __|      \n\
                               /\__/ / |_| | |_) |  __/ |    | |  | | (_| | |  | | (_) | | |_/ / | | (_) \__ \      \n\
                               \____/ \__,_| .__/ \___|_|    \_|  |_/\__,_|_|  |_|\___/  \____/|_|  \___/|___/      \n\
                                           | |                                                                      \n\
                                           |_|                                                                      ")
    printStars("\n")
    print("                                             Info: Das Skript ist mit STRG + C (je nach Terminal) zu beenden.\n")
    printStars()

def printTrainingMode():
    """Gibt den Header für den Training-Mode aus"""
    print("                                                       _______        _       _\n\
                                                      |__   __|      (_)     (_)            \n\
                                                         | |_ __ __ _ _ _ __  _ _ __   __ _ \n\
                                                         | | '__/ _` | | '_ \| | '_ \ / _` |\n\
                                                         | | | | (_| | | | | | | | | | (_| |\n\
                                                         |_|_|  \__,_|_|_| |_|_|_| |_|\__, |\n\
                                                                                       __/ |\n\
                                                                                      |___/ ")
    printStars("\n")

def printTestingMode():
    """Gibt den Header für den Testing-Mode aus"""
    print("                                                             _______        _   _             \n\
                                                        |__   __|      | | (_)            \n\
                                                           | | ___  ___| |_ _ _ __   __ _ \n\
                                                           | |/ _ \/ __| __| | '_ \ / _` |\n\
                                                           | |  __/\__ \ |_| | | | | (_| |\n\
                                                           |_|\___||___/\__|_|_| |_|\__, |\n\
                                                                                     __/ |\n\
                                                                                    |___/ ")
    printStars("\n")

def printStars(add = ""):
    """Gibt eine Zeile voller Sterne aus"""
    print("***********************************************************************************************************************************************{}".format(add))

def printProjektInfos():
    """Gibt mehrere Zeilen mit Projekt-Infos aus"""
    print("\n\tAutor: Jan Gaida" +\
    "\n\tProjekt: Deep-Q-Learning mit 'Super Mario Bros' und A3C" +\
    "\n\tGithub: https://github.com/JanGaida/research_project_machine_learning_hshof_sose2020" +\
    "\n\tVersion: 1.0.0-r" +\
    "\n\n\tCopyright (c) 2020 Jan Gaida\n")
