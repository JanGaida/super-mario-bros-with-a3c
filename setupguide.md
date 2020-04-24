---

 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Logo_fh_hof.svg/2000px-Logo_fh_hof.svg.png" width="350">

---

# Research Project: Q-Learning mit 'Super Mario Bros.'

*Seminararbeit der Vorlesung **Angewandtes Maschinelles Lernen** an der **Hochschule für angewande Wissenschaften Hof** des **Sommersemesters 2020**.*

---

## Anforderungen
- Ubunbtu 20.04 (frühere Versionen erfordern ggf. Anpassungen)
- CUDA-Kompatible GPU (<a href="https://developer.nvidia.com/cuda-gpus">Klicke hier für eine Liste der unterstützten GPU's</a>)

---

## Installations-Guide

### 1. Hinzufügen des NVIDIA-Package-Repos
```
mkdir ~/tmp && cd ~/tmp

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo apt-get update

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt-get update
```


### 2. Installieren der NVIDIA-Treiber
```
sudo apt-get install --no-install-recommends nvidia-driver-418

sudo apt-get update
```

*Anschließend den Computer **neustarten**.*

*Wenn alles geklappt habt gibt nachfolgender Befehl Informationen über die mit dem System verbundenen GPU's aus.*

```
nvidia-smi
```

### 3. Installieren der Runtime-Libraries
```
sudo apt-get install --no-install-recommends cuda-10-0 libcudnn7=7.6.2.24-1+cuda10.0 libcudnn7-dev=7.6.2.24-1+cuda10.0
```

### 4. [Optional] Installieren von TensorRT
```
sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 libnvinfer-dev=5.1.5-1+cuda10.0
```

### 5. Installieren von Anaconda
```
cd ~/tmp

wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh

bash ./Anaconda3-2019.07-Linux-x86_64.sh
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
~/anaconda3/bin/python
~/anaconda3/bin/pip
```

### 6. Installieren von Jupyter-Notebook mit Anaconda

```
conda install -c anaconda jupyter
```

### 7. Installieren von Tensorflow 2.0 mit Pip
```
pip install tensorflow-gpu==2.0.0
```

### 8. [Optional] Remote-Verbindung konfigurieren

### 9. Finale Schritte

```
cd ~

rm -r tmp

sudo apt-get update
```

---
<a href="https://medium.com/@birkann/install-tensorflow-2-0-with-gpu-support-and-jupyter-notebook-db0eeb3067a1"><font size="5">Resource</font></a>

