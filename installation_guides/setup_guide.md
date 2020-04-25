---

 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Logo_fh_hof.svg/2000px-Logo_fh_hof.svg.png" width="350">

---

# Research Project: Q-Learning mit 'Super Mario Bros.'

*Seminararbeit der Vorlesung **Angewandtes Maschinelles Lernen** an der **Hochschule für angewande Wissenschaften Hof** des **Sommersemesters 2020**.*

---

## Anforderungen
- Ubunbtu 20.04, 64-bit (andere Versionen erfordern ggf. kleinere Anpassungen)
- CUDA-Kompatible GPU (<a href="https://developer.nvidia.com/cuda-gpus">Klicke hier für eine Liste der unterstützten GPU's</a>)

---

## Installations-Guide

#### 1. Hinzufügen des NVIDIA-Package-Repos
```
mkdir ~/tmp && cd ~/tmp

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt-get update
```


### 2. Installieren der NVIDIA-Treiber
```
sudo apt-get install -y --no-install-recommends nvidia-driver-418

sudo apt-get update
```

*Anschließend den Computer **neustarten**.*

```
reboot
````

*Wenn alles geklappt habt gibt nachfolgender Befehl Informationen über die mit dem System verbundenen GPU's aus.*

```
nvidia-smi
```

#### 3. Installieren der Runtime-Libraries
```
sudo apt-get install -y --no-install-recommends cuda-10-0 libcudnn7=7.6.2.24-1+cuda10.0 libcudnn7-dev=7.6.2.24-1+cuda10.0

sudo apt-get update
```


*In den aufgelistetet Informationen findet sich der Unterpunkt 'CUDA Information', bei welchen die GPU's aufgelistet sind gefolgt von einem '[SUPPORTED]' bei erfolgreichem einrichten.*

#### 4. [Optional] Installieren von TensorRT
```
sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 libnvinfer-dev=5.1.5-1+cuda10.0

sudo apt-get update
```

#### 5. Installieren von Anaconda
```
cd ~/tmp

wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

bash ./Anaconda3-2020.02-Linux-x86_64.sh

conda update -n base -c defaults conda
```

*Bestätigen Sie anschließend die Lizenzbedingungen mit 'yes', wählen Sie das Installationsverzeichnis (Default ist '~/anaconda3') und initlialisieren Sie Anaconda3 mit 'yes'.*

*Laden Sie anschließend die bashrc-Datei neu.*

```
source ~/.bashrc
```

*Nachdem die Installation abgeschlossen ist können sie das automatische aktivieren von der Base-Anaconda-Umgebung mit folgenden Kommando deaktivieren.*

```
conda config --set auto_activate_base false

source ~/.bashrc
```

*Überprüfen Sie nun ob die Insallations geglückt ist mit folgenden Befehl:*

```
which python && which pip
```

*Das erwartete Ergebnis lautet:*
```
/home/{username}/anaconda3/bin/python
/home/{username}/anaconda3/bin/pip
```

*Sie können nun überprüfen ob CUDA korrekt installiert wurde und ordnungsgemäß funktioniert.*

```
numba -s
```

#### 6. Installieren von Jupyter-Notebook mit Anaconda

```
conda install -c anaconda jupyter
```

#### 7. Installieren von Tensorflow mit Pip
```
pip install --upgrade pip

pip install tensorflow
```

#### 8. [Optional] Remote-Verbindung konfigurieren

```
jupyter notebook --generate-config
```

```
sudo nano ~/.jupyter/jupyter_notebook_config.py
```

*Passen Sie ggf. die erstellte Config-Datei an (bspw. am Anfang der Datei). Nachfolgend ist ein Vorschlag an Änderungen für das benutzten in einem lokalen Heim-Netzwerk (!):*

```
c.NotebookApp.ip = '*' # Listen to all ip's
c.NotebookApp.notebook_dir = '/Absolute/Path/'
c.NotebookApp.port = 8888 # Use given port
c.NotebookApp.token = '' # Skip authentication
c.NotebookApp.allow_origin = '*' # Allow access from anywhere
c.NotebookApp.disable_check_xsrf = True # Allow cross-site requests
c.NotebookApp.open_browser = False # block browser from launching with jupyter
```

*Mit erwähnten Konfigurationen können Sie anschließend Jupyter-Notebook starten.*

```
jupyter notebook
```

*Anschließend können Sie sich mit einem Browser, der <b>lokalen IP</b> des Host-Rechners, sowie dem spezifizierten Port mit dem Jupyter-Notebook verbinden. (Bspw. https://localhost:8888)*

#### 9. Installieren von Git

```
sudo apt-get install -y git-all
```

*Nun muss Git noch konfiguriert werden, ersetzen Sie hierbei den <b>NAME</b> sowie <b>EMAIL</b> passend.*

```
git config --global user.name 'NAME'

git config --global user.email 'EMAIL'
```

#### 10. [Optional] Installieren und konfigurieren von SSH
```
sudo apt install -y ssh

sudo systemctl enable --now ssh
```

*Überprüfen Sie anschließend den Status des SSH-Serices:*

```
sudo systemctl status ssh
```

*Sie können sich nun mit SSH zu dem Host-Computer verbinden.*

*Um den SSH-Service automatisch zu starten geben Sie folgenden Befehl ein:*

```
sudo update-rc.d ssh defaults
```

#### 11. Finale Schritte

*Fast geschaft ...*

```
cd ~ && rm -r tmp

sudo apt-get update
```

---

<a href="https://github.com/JanGaida/research_project_machine_learning_hshof_sose2020/blob/master/styling_guide.md">Hier geht es mit dem optional stylen von Jupyter-Notebook weiter...</a>

---
<font size="5"><a href="https://medium.com/@birkann/install-tensorflow-2-0-with-gpu-support-and-jupyter-notebook-db0eeb3067a1">Quelle</a> (Diverse Anpassungen wurden vorgenommen)</font>

