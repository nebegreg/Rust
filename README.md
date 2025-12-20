# Ultimate Rotoscopy - Minimal Version

**UN fichier. UNE tâche. Ça MARCHE.**

## Installation (2 étapes)

```bash
# 1. Installer SAM3
pip install git+https://github.com/facebookresearch/sam3.git

# 2. Authentifier HuggingFace
huggingface-cli login
# Entrer votre token
```

## Usage

```bash
python roto.py image.jpg 100,100 200,200 --output mask.png
```

**C'est tout.**

### Interface PySide6 (GUI)

```bash
pip install PySide6 opencv-python numpy torch
pip install git+https://github.com/facebookresearch/sam3.git

python rotoscope_gui.py
```

1. Ouvrez une image (PNG/JPG/TIF)
2. Cliquez pour placer des points foreground
3. Choisissez le périphérique (Auto, CUDA, CPU)
4. Lancez « Segmenter » puis « Exporter le masque »

## Exemples

```bash
# Segmenter avec 1 point
python roto.py photo.jpg 150,200

# Segmenter avec 3 points
python roto.py photo.jpg 100,100 200,150 300,200

# Sur CPU (si pas de GPU)
python roto.py photo.jpg 100,100 --device cpu

# Les points doivent être dans l'image (x,y >= 0)
python roto.py photo.jpg 120,80 200,140
```

## Ça fait quoi?

1. Charge SAM3
2. Lit votre image
3. Segmente l'objet aux points donnés
4. Sauvegarde `mask.png`

**Pas de GUI. Pas de vidéo. Pas de complexité. Juste ça.**

## Prochaines étapes (SI ça marche)

Une fois que ce script fonctionne:

1. ✅ Ajouter Depth Anything V3
2. ✅ Ajouter MatAnyone
3. ✅ Traiter plusieurs images
4. ✅ Faire une GUI simple
5. ✅ Export professionnel

**Mais d'abord, ce script doit marcher.**

## Troubleshooting

**SAM3 not installed**
```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

**Authentication error**
```bash
huggingface-cli login
# Créer token sur: https://huggingface.co/settings/tokens
```

**No GPU**
```bash
python roto.py image.jpg 100,100 --device cpu
```

**CUDA non dispo (fallback auto)**
```bash
# Demande CUDA mais basculera sur CPU si indisponible
python roto.py image.jpg 100,100 --device cuda
```

## Tester

```bash
# Télécharger image test
wget https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg

# Segmenter le camion (point au centre)
python roto.py truck.jpg 400,300

# Vérifier mask.png créé
ls -lh mask.png
```

---

**C'est simple. C'est ce qu'on veut.**
