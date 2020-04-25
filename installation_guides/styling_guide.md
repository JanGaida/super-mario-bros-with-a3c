---

 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Logo_fh_hof.svg/2000px-Logo_fh_hof.svg.png" width="350">

---

# Research Project: Q-Learning mit 'Super Mario Bros.'

*Seminararbeit der Vorlesung **Angewandtes Maschinelles Lernen** an der **Hochschule für angewande Wissenschaften Hof** des **Sommersemesters 2020**.*

---

## Anforderungen
- Ubunbtu 20.04, 64-bit (andere Versionen erfordern ggf. kleinere Anpassungen)
- CUDA-Kompatible GPU (<a href="https://developer.nvidia.com/cuda-gpus">Klicke hier für eine Liste der unterstützten GPU's</a>)
- Abgeschlossen Cuda- und Jupyter-Notebook-Installation (<a href="https://github.com/JanGaida/research_project_machine_learning_hshof_sose2020/blob/master/setup_guide.md">Klicke hier für eine konforme Anleitung</a>)
---

<b>!!!</b> *Nachfolgend sind <b>optionale</b> Anpassungen an dem bereits installierten Jupyter-Notebook*

---

#### 1. Installieren von Jupyter-Erweiterungen

*Um Erweiterung zu installieren wird das Pip-Package 'jupyter_contrib_nbextensions' benötigt, zu erst benötigten wir hierfür ein Terminal - auf das in Jupyter-Notebook integrierte greift man folgend zu:*

```
Home -> New -> Terminal
```

*Anschließend können folgende Befehle ausgeführt werden:*

```
pip install jupyter_contrib_nbextensions

jupyter contrib nbextension install --user

jupyter nbextensions_configurator enable --user
```

*Nun muss das Jupyter-Notebook neugestartet werden, dazu geben Sie im Terminal des Host-Computers folgende Befehle ein:*

```
jupyter notebook stop

jupyter notebook
```

*Anschließend muss im Browser die Website des Jupyter-Notebooks neugeladen werden. Auf der Startseite gibt es nun einen neuen Reiter 'Nbextensions'.*

#### 2. [Optional] Strukturierung

*Nachfolgend ist beschrieben wie eine Jupyterweiterung zur strukturierten Darstellung im Jupyter-Notebook aktiviert wird. Hierfür kann die zuvor installierte Erweiterung 'Nbextensions' genutzt werden:*

```
Home -> Nbextensions
```

*Zunächst muss der Hacken bei <b>'disable configuration for nbextensions without explicit compatibility [...]'</b> entfernt werden.*

*Anschließend kann nach <b>'collapsible headings'</b> gesucht werden und diese Erweiterung mit einem Klick aktiviert werden.*

*Lädt man nun die offnen Jupyter-Notebooks neu so werden diese mit der Erweiterung geladen.*

#### 3. [Optional] Ausführungszeiten

*Nachfolgend ist beschrieben wie eine Jupyterweiterung zur Darstellung von Ausführungszeiten im Jupyter-Notebook installiert wird. Hierfür kann die zuvor installierte Erweiterung 'Nbextensions' genutzt werden*:

```
Home -> Nbextensions
```

*Zunächst muss der Hacken bei <b>'disable configuration for nbextensions without explicit compatibility [...]'</b> entfernt werden.*

*Anschließend kann nach <b>'executetime'</b> gesucht werden und diese Erweiterung mit einem Klick aktiviert werden.*

*Lädt man nun die offnen Jupyter-Notebooks neu so werden diese mit der Erweiterung geladen.*

#### 4. [Optional] Inhaltsverzeichnis

*Nachfolgend ist beschrieben wie eine Jupyterweiterung zur Darstellung von Ausführungszeiten im Jupyter-Notebook installiert wird. Hierfür kann die zuvor installierte Erweiterung 'Nbextensions' genutzt werden*:

```
Home -> Nbextensions
```

*Zunächst muss der Hacken bei <b>'disable configuration for nbextensions without explicit compatibility [...]'</b> entfernt werden.*

*Anschließend kann nach <b>'table of contents'</b> gesucht werden und diese Erweiterung mit einem Klick aktiviert werden.*

*Lädt man nun die offnen Jupyter-Notebooks neu so werden diese mit der Erweiterung geladen.*

#### 5. [Optional] Theme

*Nachfolgende ist beschrieben wie eine Jupyterweiterung zum Anpassen des Jupyter-Themes installiert wird. Zunächst muss mit Pip das 'jupyterthemes'-Package benötigt, dazu wird zu erst ein Terminal gebraucht:*

```
Home -> New -> Terminal
```

*Anschließend kann das Package installiert werden:*

```
pip install jupyterthemes
```

*Nun kann das gewünschte Theme aktivert werden, zur Auswahl stehen folgende Themes:*
- *onedork [empfohlen]*
- *grade3*
- *oceans16*
- *chesterish*
- *monokai*
- *solarizedl*
- *solarizedd*

*Screenshots der aufgeführten Themes sind <a href="https://github.com/dunovank/jupyter-themes/tree/master/screens">hier</a> zufinden.*

*Kommando zum aktivieren des Themes:*
```
jt -t <theme-name>
```

*Kommando zum zurücksetzen des Themes:*
```
jt -r
```

*Um Änderungen am Theme zu übernehmmen muss ggf. das Jupyter-Notebook neugestartet werden.*
```
jupyter notebook stop

jupyter notebook
```

---

<a href="https://github.com/JanGaida/research_project_machine_learning_hshof_sose2020">Hier geht es zurück zum Research-Projekt</a>

---
