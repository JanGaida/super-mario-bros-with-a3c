[![Generic badge](https://img.shields.io/badge/License-Properitary-red.svg)](https://github.com/JanGaida/research_project_machine_learning_hshof_sose2020/blob/master/LICENSE.md)
[![Active Development](https://img.shields.io/badge/Maintenance%20Level-Actively%20Developed-brightgreen.svg)](https://github.com/JanGaida/research_project_machine_learning_hshof_sose2020/)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/JanGaida/research_project_machine_learning_hshof_sose2020/issues)
[![Python 3.7.7](https://img.shields.io/badge/python-3.7.7-blue.svg)](https://www.python.org/downloads/release/python-377/)
[![JupyterNotebook 6.03](https://img.shields.io/badge/Jupyter_Notebook-6.0.3-orange.svg)](https://jupyter.org/)

---

 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Logo_fh_hof.svg/2000px-Logo_fh_hof.svg.png" width="350">

---

# Research Project: Q-Learning mit 'Super Mario Bros.'

*Seminararbeit der Vorlesung **Angewandtes Maschinelles Lernen** an der **Hochschule für angewande Wissenschaften Hof** des **Sommersemesters 2020**.*

---


## 1. Anleitung:

Nachfolgend eine kurze Anleitung zum verwenden dieses Projekt.


### 1.1 Docker-Setup

Nachfolgend eine kurze Erklärung wie dieses Notebook mit Docker ausgeführt werden kann.

Eine funktionierende Docker-Installation wird hierfür benötigt ([s.h. Docker-Dokumentation](https://docs.docker.com/docker-for-windows/install/))

#### 1.1.1 Herunterladen des Images

Herunterladen des [Dockerimages 'jupyter/tensorflow-notebook'](https://hub.docker.com/r/jupyter/tensorflow-notebook):
```
$ docker pull jupyter/tensorflow-notebook
```

#### 1.1.2 Starten des Containers

Nachdem das Image heruntergeladen ist kann der Container gebaut und gestartet werden.

```
$ docker run -d -p 8888:8888 jupyter/tensorflow-notebook
```

Wenn der Container bereits gestartet worden ist:

```
$ docker start -i <container-id>
```

#### 1.1.3 Jupyter-Access-Token

Zunächst wird die Container-Id benötigt, dazu können alle laufenden Container in einer List aufgeführt werden:

```
$ docker ps
```

Anschließend wird ein Kommando ausgeführt um das Jupyter-Access-Token auszulesen (hierfür wird die Container-Id aus dem Schritt vorher benötigt und ist anstelle '<container-id>' einzusetzen).

```
$ docker exec <container-id> jupyter notebook list
```

Zurückgegeben wird eine Link zu den Localhost an den definierten Port mit Token als Parameter. Dieser Link kann nun in einem Browser geöffnet werden (ggf. muss die Adresse des Localhost angepasst werden, z.b. '0.0.0.0' mit 'localhost' ausgetauscht werden o.ä.).


### 1.2 Jupyter-Notbook-Setup

Nachfolgend sind <b>optionale</b> Anpassungen an dem im Container installierten Jupyter-Notebook.

Um Erweiterung zu installieren wird das Pip-Package 'jupyter_contrib_nbextensions' benötigt, zu erst benötigten wir hierfür ein Terminal - auf das in Jupyter-Notebook integrierte greift man folgend zu:

```
Home -> New -> Terminal
```

Anschließend können folgende Befehle ausgeführt werden:

```
$ pip install jupyter_contrib_nbextensions
$ jupyter contrib nbextension install --user
$ jupyter nbextensions_configurator enable --user
```

Nun muss der Docker-Container neugestartet werden:

```
$ docker stop <container-id>
$ docker start -i <container-id>
```

Anschließend wird ein Link zu dem Jupyter-Notebook ausgegeben und auf der Startseite gibt es nun einen neuen Reiter 'Nbextensions'.

#### 1.2.1 Strukturierung

Nachfolgend ist beschrieben wie eine Jupyterweiterung zur strukturierten Darstellung im Jupyter-Notebook installiert wird.

Hierfür kann die zuvor installierte Erweiterung 'Nbextensions' genutzt werden:

```
Home -> Nbextensions
```

Zunächst muss der Hacken bei *'disable configuration for nbextensions without explicit compatibility [...]'* entfernt werden.

Anschließend kann nach *'Collapsible Headings'* gesucht werden und diese Erweiterung mit einem Klick aktiviert werden.

Lädt man nun die offnen Jupyter-Notebooks neu so werden diese mit der Erweiterung geladen.


#### 1.2.2 Ausführungszeit

Nachfolgend ist beschrieben wie eine Jupyterweiterung zur Darstellung von Ausführungszeiten im Jupyter-Notebook installiert wird.

Hierfür kann die zuvor installierte Erweiterung 'Nbextensions' genutzt werden:

```
Home -> Nbextensions
```

Zunächst muss der Hacken bei *'disable configuration for nbextensions without explicit compatibility [...]'* entfernt werden.

Anschließend kann nach *'ExecuteTime'* gesucht werden und diese Erweiterung mit einem Klick aktiviert werden.

Lädt man nun die offnen Jupyter-Notebooks neu so werden diese mit der Erweiterung geladen.

#### 1.2.3 Theme

Nachfolgende ist beschrieben wie eine Jupyterweiterung zum Anpassen des Jupyter-Themes installiert wird.

Zunächst muss mit Pip das 'jupyterthemes'-Package benötigt, dazu wird zu erst ein Terminal gebraucht:

```
Home -> New -> Terminal
```

Anschließend kann das Package installiert werden:

```
$ pip install jupyterthemes
```

Nun kann das gewünschte Theme aktivert werden, zur Auswahl stehen folgende Themes:
- onedork
- grade3
- oceans16
- chesterish
- monokai
- solarizedl
- solarizedd

Kommando zum aktivieren des Themes:
```
$ jt -t <theme-name>
```

Kommando zum zurücksetzen des Themes:
```
$ jt -r
```

Gegebenenfalls muss der Docker-Container neugestartet werden:

```
$ docker stop <container-id>
$ docker start -i <container-id>
```

---

***Readme in Überarbeitung..***
