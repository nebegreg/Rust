# Installation Rapide - Ultimate Rotoscopy v3 (SAM3 + DA3)

## 📦 Étape 1: Nettoyage

```bash
# Désinstallez l'ancienne version
pip uninstall ultimate_rotoscopy -y

# Nettoyez les anciennes installations
find /home/reepost -name "ultimate_rotoscopy" -type d 2>/dev/null
# Supprimez ou renommez les anciennes copies
```

## 📥 Étape 2: Extraction

```bash
cd /home/reepost/
unzip ultimate_rotoscopy_fixed.zip -d ultimate_rotoscopy_v3
cd ultimate_rotoscopy_v3
```

## 🔧 Étape 3: Installation

```bash
# Activez votre environnement virtuel
source /path/to/venv/bin/activate

# Installez les dépendances
pip install -r requirements.txt

# Installez le package en mode développement
pip install -e .

# Si vous avez du code Rust (optionnel):
# maturin develop --release
```

## 🎯 Étape 4: Installez SAM3 et Depth Anything V3

**IMPORTANT**: Cette version nécessite SAM3 et DA3 (pas v2!)

```bash
# SAM3 (nécessite Python 3.12+, PyTorch 2.7+, CUDA 12.6+)
pip install sam3

# Depth Anything V3 (nécessite Python 3.10+, PyTorch 2.0+, CUDA 11.8+)
pip install depth-anything-3

# MatAnyone (dernière version pour vidéo)
pip install matanyone
```

## ✅ Étape 5: Vérification

```bash
# Testez l'installation
python test_installation.py

# Si tout est OK, lancez l'application
rotoscopy-gui
```

## 🐛 Corrections Appliquées dans ce Build

### 1. **Architecture v3-only (BREAKING CHANGE)**
- ❌ Supprimé: Tous les fallbacks vers SAM2.1/SAM1
- ❌ Supprimé: Tous les fallbacks vers Depth Anything V2
- ✅ Ajouté: Messages d'erreur clairs si SAM3/DA3 non installés
- ✅ Ajouté: Dropdowns GUI montrent "SAM3" et "DA3" (pas v2!)

**Raison**: SAM2/SAM3 et DA2/DA3 ont des APIs incompatibles. Les mélanger causait des crashes.

### 2. **Fix Q_ARG Marshalling Error**
- ❌ Supprimé: `QMetaObject.invokeMethod` avec `Q_ARG(object, request)`
- ✅ Ajouté: Signal Qt natif `process_requested` pour communication thread-safe

**Raison**: PySide6 ne peut pas marshaller dataclass Python via Q_ARG → RuntimeError

### 3. **Fix process_sync() Returns None**
- ✅ Ajouté: QEventLoop pour blocage synchrone propre
- ✅ Ajouté: Gestion d'erreur avec disconnect propre des signaux

**Raison**: La méthode retournait None immédiatement sans attendre le worker thread

## 📋 Fichiers Principaux Modifiés

```
src/ultimate_rotoscopy/gui/backend.py:
  - Ligne 680: Ajout signal process_requested
  - Ligne 698: Connexion signal → worker.process
  - Ligne 815: Émission signal au lieu de Q_ARG
  - Ligne 169-203: SAM3-only avec erreur claire
  - Ligne 205-239: DA3-only avec erreur claire
  - Supprimé: ~200 lignes de code fallback v2

src/ultimate_rotoscopy/gui/main_window.py:
  - Ligne 393: Titre "Segmentation (SAM3)"
  - Ligne 398: Dropdown "SAM3 Large/Base/Small/Tiny"
  - Ligne 452: Titre "Depth Estimation (DA3)"
  - Ligne 457: Dropdown "Depth Anything V3 Large/Base/Small/Giant"
```

## 🎮 Utilisation

### GUI
```bash
rotoscopy-gui
```

### CLI
```bash
# Segmentation
rotoscopy segment image.jpg --output mask.png --points 100,100 200,200

# Matting
rotoscopy matte image.jpg --mask mask.png --output alpha.png

# Depth
rotoscopy depth image.jpg --output depth.exr
```

## 🔍 Troubleshooting

### Erreur: "SAM3 not installed"
```bash
pip install sam3
# Nécessite: Python 3.12+, PyTorch 2.7+, CUDA 12.6+
```

### Erreur: "Depth Anything V3 not installed"
```bash
pip install depth-anything-3
# Nécessite: Python 3.10+, PyTorch 2.0+, CUDA 11.8+
```

### Erreur: "RuntimeError: Q_ARG marshalling"
→ Vous utilisez une vieille version! Réinstallez depuis ce zip.

### Warnings Gtk "Failed to measure available space"
→ Ignorez-les, c'est un problème Rocky Linux + PySide6, pas critique.

## 📊 Configuration Système Recommandée

- **OS**: Rocky Linux 9 (ou RHEL-based)
- **GPU**: NVIDIA RTX 4090 (ou RTX 30xx/40xx avec 12+ GB VRAM)
- **Python**: 3.12+ (pour SAM3) ou minimum 3.10
- **PyTorch**: 2.7+ (pour SAM3) avec CUDA 12.6+
- **RAM**: 32 GB+ recommandé
- **Stockage**: 50 GB+ pour les modèles

## 📞 Support

Si vous avez des erreurs, lancez:
```bash
python test_installation.py > install_report.txt 2>&1
```

Et partagez le fichier `install_report.txt` pour diagnostic.

## 🎉 Ready to Roto!

Une fois l'installation réussie:
1. Lancez `rotoscopy-gui`
2. Load une image
3. Ajoutez des points (clic gauche = foreground, clic droit = background)
4. Clic "Generate Mask"
5. Enjoy! 🚀
