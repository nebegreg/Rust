# Ultimate Rotoscopy - SAM3 + Depth Anything V3

**Architecture Simple et Propre** - Pas de fallbacks, pas de mélange v2/v3

## 🎯 Objectif

Application de rotoscopy professionnelle utilisant:
- **SAM3** (Meta) - Segmentation de pointe
- **Depth Anything V3** (ByteDance) - Estimation de profondeur
- **Architecture propre** - Un seul venv, zéro conflit

## 📋 Prérequis

- **OS**: Rocky Linux 9 (ou RHEL-based)
- **GPU**: NVIDIA RTX 4090 (ou RTX 30xx/40xx avec 12+ GB VRAM)
- **Python**: 3.12+
- **CUDA**: 12.6+ (vérifier avec `nvidia-smi`)
- **Compte HuggingFace**: Requis pour SAM3

## 🚀 Installation (Simple)

### Étape 1: Préparer HuggingFace

SAM3 nécessite authentification:

```bash
# 1. Créer compte sur https://huggingface.co/join
# 2. Demander accès: https://huggingface.co/facebook/sam3
# 3. Créer token: https://huggingface.co/settings/tokens (type: Read)
```

### Étape 2: Installation Automatique

```bash
chmod +x install_sam3_da3.sh
./install_sam3_da3.sh
```

Le script va:
1. ✅ Vérifier Python 3.12 et GPU
2. ✅ Créer venv propre
3. ✅ Installer PyTorch 2.7+ avec CUDA 12.6
4. ✅ Installer Depth Anything V3
5. ✅ Authentifier HuggingFace
6. ✅ Installer SAM3
7. ✅ Vérifier compatibilité

### Étape 3: Test

```bash
source venv_ultimate/bin/activate
python test_models.py
```

Vous devriez voir:
```
✓ SAM3 loaded in 5.2s
✓ Depth Anything V3 loaded in 3.8s
✓ SUCCESS - Both models working!
```

## 📁 Structure du Projet

```
.
├── install_sam3_da3.sh           # Script d'installation
├── test_models.py                # Test de compatibilité
├── ROADMAP_CLEAN.md              # Plan de développement
├── COMPATIBILITY_ANALYSIS.md     # Analyse de compatibilité
│
├── venv_ultimate/                # Environnement virtuel (après install)
│
└── (à venir)
    ├── examples/
    │   ├── demo_sam3.py          # Demo SAM3 seul
    │   ├── demo_depth.py         # Demo DA3 seul
    │   └── demo_combined.py      # Les deux ensemble
    │
    └── src/
        └── ultimate_roto/
            ├── sam3_wrapper.py   # Wrapper SAM3 simple
            ├── depth_wrapper.py  # Wrapper DA3 simple
            └── pipeline.py       # Pipeline combiné
```

## 🧪 Utilisation Basique

### SAM3 - Segmentation

```python
from sam3 import sam3_model_registry, Sam3ImagePredictor
import numpy as np
import cv2

# Charger image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Charger SAM3
sam_checkpoint = sam3_model_registry["sam3_hiera_large"]()
predictor = Sam3ImagePredictor(sam_checkpoint)

# Segmenter avec points
predictor.set_image(image)
points = np.array([[100, 100], [200, 200]])  # Points foreground
labels = np.array([1, 1])  # 1=foreground, 0=background

masks, scores, logits = predictor.predict(
    point_coords=points,
    point_labels=labels,
    multimask_output=True
)

# Best mask
best_mask = masks[scores.argmax()]
cv2.imwrite("mask.png", (best_mask * 255).astype(np.uint8))
```

### Depth Anything V3 - Depth

```python
from depth_anything_3 import DepthAnythingV3
import numpy as np
import cv2

# Charger image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Charger DA3
depth_model = DepthAnythingV3.from_pretrained(
    "depth-anything/Depth-Anything-V3-Large",
    device="cuda"
)

# Estimer depth
depth = depth_model.infer_image(image)

# Normaliser et sauvegarder
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
depth_vis = (depth_normalized * 255).astype(np.uint8)
cv2.imwrite("depth.png", depth_vis)
```

## 🐛 Troubleshooting

### Problème: SAM3 import fails

```
ImportError: No module named 'sam3'
```

**Solution**:
```bash
# Vérifier authentification HuggingFace
huggingface-cli whoami

# Si pas authentifié:
huggingface-cli login
# Entrer votre token

# Réinstaller SAM3
pip install --force-reinstall git+https://github.com/facebookresearch/sam3.git
```

### Problème: DA3 import fails

```
ImportError: No module named 'depth_anything_3'
```

**Solution**:
```bash
pip install --force-reinstall git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

### Problème: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Utiliser modèles plus petits:
   - SAM3: `sam3_hiera_small` au lieu de `large`
   - DA3: `Depth-Anything-V3-Small` au lieu de `Large`

2. Traiter images plus petites:
   ```python
   # Resize avant traitement
   image = cv2.resize(image, (1280, 720))
   ```

3. Libérer mémoire entre traitements:
   ```python
   torch.cuda.empty_cache()
   ```

### Problème: PyTorch version incompatible

```
ImportError: torch version mismatch
```

**Solution**:
```bash
# Réinstaller PyTorch 2.7+
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 📊 Performance Attendue (RTX 4090)

- **SAM3 Large**: ~5s chargement, ~0.5s/image
- **DA3 Large**: ~4s chargement, ~0.3s/image
- **Mémoire GPU**: ~8GB pour les deux modèles chargés

## 🎯 Prochaines Étapes

1. **Phase 1 ✅**: Installation et tests → **VOUS ÊTES ICI**
2. **Phase 2**: Wrappers Python simples
3. **Phase 3**: Pipeline combiné
4. **Phase 4**: GUI minimale (optionnel)
5. **Phase 5**: Package installable

## 📚 Ressources

- SAM3 Repo: https://github.com/facebookresearch/sam3
- DA3 Repo: https://github.com/ByteDance-Seed/Depth-Anything-3
- SAM3 HuggingFace: https://huggingface.co/facebook/sam3

## ❓ Questions Fréquentes

**Q: Pourquoi Python 3.12 exactement?**
A: SAM3 nécessite 3.12+, DA3 accepte jusqu'à 3.13. Python 3.12 est le sweet spot.

**Q: Puis-je utiliser CPU au lieu de GPU?**
A: Oui, mais 100x plus lent. SAM3 nécessite GPU pour être pratique.

**Q: SAM3 vs SAM2.1, quelle différence?**
A: SAM3 a text prompting, meilleure qualité, plus de paramètres (848M vs 600M).

**Q: Depth Anything V3 vs V2?**
A: DA3 a multi-view, camera estimation, 3D Gaussian splatting. API différente!

## 📝 License

- SAM3: Apache 2.0 (Meta)
- Depth Anything V3: Apache 2.0 (ByteDance)
