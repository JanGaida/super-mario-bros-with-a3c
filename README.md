<p align="center">
  <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/LICENSE.md"><img alt="Licence" src="https://img.shields.io/badge/License-Properitary-red.svg"></a>
  <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/pulse"><img alt="Active Development" src="https://img.shields.io/badge/Maintenance%20Level-Actively%20Developed-brightgreen.svg"></a>
  <a href="https://docs.python.org/3/"><img alt="Python 3.8" src="https://img.shields.io/badge/Python-3.8-blue.svg"></a>
  <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/issues"><img alt="Ask Me Anything" src="https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg"></a>
  <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/commits/master"><img alt="last commit" src="https://img.shields.io/github/last-commit/JanGaida/super-mario-bros-with-a3c?style=flat&logo=GitHub"></a>
  <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/JanGaida/super-mario-bros-with-a3c?style=flat&logo=GitHub"></a>
</p>

---

<p align="center">
<a href="https://www.hof-university.de/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Logo_fh_hof.svg/2000px-Logo_fh_hof.svg.png" width="250"></a>
</p>

#### Student Research Project: 
# Deep-Q-Learning mit 'Super Mario Bros' und A3C.

*Seminararbeit der Vorlesung **Angewandtes Maschinelles Lernen** an der **Hochschule für angewande Wissenschaften Hof** des **Sommersemesters 2020**.*

---

## Paper

*upcoming*

---

## Ziele

- Entwicklen, Traninern und Testen eines Neuronalen Netzes zum absovlieren von <a href="https://de.wikipedia.org/wiki/Super_Mario_Bros.">Super Mario Bros. (1985)</a>
- Verständnis von Neuronalen Netzten und RL-Algorithmen (insbesondere Deep-Q-Learning) ausbilden

---

## Installation

#### Benötigt:
- lauffähige **Python-3**-Installtion (<a href="https://docs.python.org/3/installing/index.html">Anleitung</a>)
- lauffähiges **Pip3**-Paketemanager (in Python-3.4 und später bereits enthalten)
- **CUDA-Treiber** für die Gpu-Unterstützung¹ (<a href="https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html">Anleitung</a>)

#### Repository clonen:
Laden Sie sich eine Kopie dieses Projektes lokal herunter, hier bspw. mit <a href="https://git-scm.com/docs/git">Git</a>:

```shell
cd <destination>
git clone https://github.com/JanGaida/super-mario-bros-with-a3c.git
```

#### Dependencies installieren:
Installieren Sie anschließend die benötigten Pakete:

*Für **Gpu**-Unterstützung*¹:
```shell
cd <destination>/super-mario-bros-with-a3c
pip3 install tensorflow-gpu==2.2.0
pip3 install -r requirements.txt
```

*Für **Cpu**-Unterstützung*:
```shell
cd <destination>/super-mario-bros-with-a3c
pip3 install tensorflow==2.2.0
pip3 install -r requirements.txt
```

¹ *für das Trainiern empfohlen; Gpu-Empfehlung: Nvidia-Gpu, GTX 9XX oder neuer, 6GB NVRAM für 5 parallel laufende Trainings-Prozesse*

---

## Starten
Das Progamm selbst ist darauf ausgelegt, dass mit diversen Parametern experimentiert werden kann und verfügt deshalb um eine große Anzahl an Start-Argumenten. Desweitern verfügt es über viele Kommentare im Code um bspw. die Netzwerkarchitektur zuändern, wie etwa das *LSTM*-Modul mit einem *GRU*-Modul auszutauschen.

#### Start-Argumente:
Für eine Auflistung aller Start-Argumente:
```shell
cd <destination>/super-mario-bros-with-a3c
python3 main.py --help
```

#### Trainieren:
Für das Starten des Training-Vorgangs des Super-Mario-Bros Level 1 in Welt 1 von 10000 Episoden mit 5 Trainingsthreads:
```shell
cd <destination>/super-mario-bros-with-a3c
# Ausführliche Schreibweise:
python3 main.py --mode training --world 1 --stage 1 --absolute_max_training_steps 10000 --num_parallel_trainings_threads 5
# Kurze Schreibweise:
python3 main.py -m training -w 1 -s 1 -max 10000 -ttr 5
```

#### Testen:
Für das Starten des Testing-Vorgangs des Super-Mario-Bros Level 1 in Welt 1:
```shell
cd <destination>/super-mario-bros-with-a3c
# Ausführliche Schreibweise:
python3 main.py --mode testing --world 1 --stage 1
# Kurze Schreibweise:
python3 main.py -m testing -w 1 -s 1
```
---

## Ergebnisse

*upcoming*

---

## Neuronales Netzwerk

**Grundlegende Architektur:**
<img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Basis_Architektur.png?raw=true" />

**Netzwerk-Architektur:**
<img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur.png?raw=true" />

---

## Reward-Funktion

*upcomming*

---

## Credits & Special Thanks

- Dem Professor und den Komilitonen der Vorlesung "Angewandtes maschinelles Lernen (Sommersemester 2020)"
- Christian Kauten für das Bereitstellen des <a href="https://github.com/Kautenja/gym-super-mario-bros">OpenAi-Gym's Super-Mario-Bros.</a> 
- <a href="https://github.com/uvipen/Super-mario-bros-A3C-pytorch">Viet Nguyen</a>, <a href="https://github.com/ArvindSoma/a3c-super-mario-pytorch">Arvind Soma</a>, <a href="https://github.com/sachinruk/Mario">Sachin Abeywardana</a>, <a href="https://github.com/Kautenja/playing-mario-with-deep-reinforcement-learning">Christian Kauten</a>,  <a href="https://github.com/sadeqa/Super-Mario-Bros-RL">Otmane Sakhi & Amine Sadeq</a> und <a href="https://github.com/search?q=super+mario+bros+reinforcement-learning&type=Repositories">viele weitere Projekte</a> für inspirerende Reinforcement-Learning-Implementationen im Super-Mario-Bros-Gym's

---
