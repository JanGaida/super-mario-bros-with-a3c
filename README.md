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


#### Welt 1 Level 1:
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_1000_x_5__dcn_w1s1v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_1000_x_5__dcn_w1s1v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_1500_x_5__dcn_w1s1v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_1500_x_5__dcn_w1s1v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_2000_x_5__dcn_w1s1v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_2000_x_5__dcn_w1s1v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_3500_x_5__dcn_w1s1v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_3500_x_5__dcn_w1s1v0.gif?raw=true" width="192"/></a></p>

<p align="center"><i>Von links nach rechts: 1k Episoden, 1.5k Episoden, 2k Episoden und 3.5k Episoden bei jeweils 5 Workern</i></p>
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/stats/stats_w1s1v0_dcn.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/stats/stats_w1s1v0_dcn.png?raw=true"/></a></p>

#### Welt 1 Level 3:
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_500_x_5__dcn_w1s3v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_500_x_5__dcn_w1s3v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_1000_x_5__dcn_w1s3v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_1000_x_5__dcn_w1s3v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_2000_x_5__dcn_w1s3v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_2000_x_5__dcn_w1s3v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_2500_x_5__dcn_w1s3v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_2500_x_5__dcn_w1s3v0.gif?raw=true" width="192"/></a></p>

<p align="center"><i>Von links nach rechts: 0.5k Episoden, 1k Episoden, 2k Episoden und 2.5k Episoden bei jeweils 5 Workern</i></p>
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/stats/stats_w1s3v0_dcn.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/stats/stats_w1s3v0_dcn.png?raw=true"/></a></p>

#### Welt 1 Level 4:
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_1000_x_5__dcn_w1s4v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_1000_x_5__dcn_w1s4v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_2000_x_5__dcn_w1s4v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_2000_x_5__dcn_w1s4v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_3000_x_5__dcn_w1s4v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_3000_x_5__dcn_w1s4v0.gif?raw=true" width="192"/></a> <a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_7500_x_5__dcn_w1s4v0.gif"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/gifs/ep_7500_x_5__dcn_w1s4v0.gif?raw=true" width="192"/></a></p>

<p align="center"><i>Von links nach rechts: 1k Episoden, 2k Episoden, 3k Episoden und 7.5k Episoden bei jeweils 5 Workern</i></p>
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/stats/stats_w1s4v0_dcn.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/stats/stats_w1s4v0_dcn.png?raw=true"/></a></p>

#### Model-Files:
Vortrainierte PyTorch-Model-Files befinden sich <a href="https://drive.google.com/drive/folders/1OnAW680HXIiHOvfb-MxD0GHxbLoGcpCq?usp=sharing">hier</a> (<a href="https://docs.google.com/document/d/1UDgxDr2wFpdXbE7-w9TowjG0rgAvxz345Ac1zRAC1v8/edit?usp=sharing">Anleitung</a>).

#### Aufnahmen:
Weitere Aufnahmen zu abgeschlossenen Trainings befinden sich <a href="https://drive.google.com/drive/folders/1Pzn9ArpvdYdisD2qADFvM7LK059E8etK?usp=sharing">hier</a>

---

## Neuronales-Netzwerk

#### Neuronales-Netzwerk-Context:
Nachfolgend ist der Context des Neuronalen-Netzwerkes zusehen.
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Basis_Architektur_c.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Basis_Architektur_c.png?raw=true"/></a></p>

#### Neuronales-Netzwerk-Architektur:
Nachfolgend sind die genauen Architekturen der Neuronalen-Netzwerkes zusehen die implemtiert wurden.

##### Naive-CNN-Ansatz:
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur_cn.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur_cn.png?raw=true" height="500"/></a></p>

##### Deep-CNN-Ansatz:
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur_dcn.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur_dcn.png?raw=true" height="500"/></a></p>

##### LSTM-RNN-Ansatz:
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur_lstm.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur_lstm.png?raw=true" height="250"/></a></p>

##### GRU-RNN-Ansatz:
<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur_gru.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/nn/Netzwerk_Architektur_gru.png?raw=true" height="250"/></a></p>

---

## Frame-Preprocessing

Nachfolgend ist das Frame-Preprocessing in einer Graphik kurz erklärt.

<p align="center"><a href="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/frames/preprocessing.png"><img src="https://github.com/JanGaida/super-mario-bros-with-a3c/blob/master/doc/frames/preprocessing.png?raw=true"/></a>

---

## Reward-Funktion

Der Reward selbst wurde als sog. 'Reward Shaping' implementiert und besteht aus vier Bestandteilen:

##### - Delta-X-Position
Formula: <i><b>delta_x</b> = x_1 - x_0</i>

##### - Delta-Time
Formula: <i><b>delta_time</b> = min( (t_1 - t_0), 0 )</i>

##### - R-Ziel
Formula: <i><b>r_ziel</b> = 45 if flag_reached else 0</i>

##### - R-Life
Formula: <i><b>r_life</b> = -45 if life_lost else 0</i>

##### → Reward

Insgesammt bildet sich der Reward schließlich nach nachfolgenden Formular:

Formula: <i><b>reward</b> = (delta_x + delta_time + r_ziel + r_life) / 10</i>


---

## Credits & Special Thanks

- Dem Professor und den Komilitonen der Vorlesung "Angewandtes maschinelles Lernen (Sommersemester 2020)"
- Christian Kauten für das Bereitstellen des <a href="https://github.com/Kautenja/gym-super-mario-bros">OpenAi-Gym's Super-Mario-Bros.</a> 
- <a href="https://github.com/uvipen/Super-mario-bros-A3C-pytorch">Viet Nguyen</a>, <a href="https://github.com/ArvindSoma/a3c-super-mario-pytorch">Arvind Soma</a>, <a href="https://github.com/sachinruk/Mario">Sachin Abeywardana</a>, <a href="https://github.com/Kautenja/playing-mario-with-deep-reinforcement-learning">Christian Kauten</a>,  <a href="https://github.com/sadeqa/Super-Mario-Bros-RL">Otmane Sakhi & Amine Sadeq</a> und <a href="https://github.com/search?q=super+mario+bros+reinforcement-learning&type=Repositories">viele weitere Projekte</a> für inspirerende Reinforcement-Learning-Implementationen im Super-Mario-Bros-Gym's

---
