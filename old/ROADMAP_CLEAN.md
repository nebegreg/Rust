# Roadmap - Ultimate Rotoscopy v3.0
# SAM3 + Depth Anything V3 Integration

## Phase 1: Environment Setup ✓
**Objectif**: Un seul venv avec SAM3 et DA3 qui fonctionnent ensemble

### Étape 1.1: Analyser les requirements
- [ ] Vérifier SAM3 requirements (https://github.com/facebookresearch/sam3)
  - Python version
  - PyTorch version
  - CUDA version
  - Dépendances

- [ ] Vérifier DA3 requirements (https://github.com/ByteDance-Seed/Depth-Anything-3)
  - Python version
  - PyTorch version
  - CUDA version
  - Dépendances

- [ ] Identifier conflits potentiels
  - Versions PyTorch compatibles?
  - Versions CUDA compatibles?
  - Dépendances communes?

### Étape 1.2: Installation ordre correct
```bash
# 1. Créer venv propre
python3.12 -m venv venv_ultimate
source venv_ultimate/bin/activate

# 2. Installer PyTorch d'abord (version compatible avec les deux)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Installer SAM3
pip install git+https://github.com/facebookresearch/sam3.git

# 4. Installer Depth Anything V3
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

# 5. Vérifier pas de conflits
pip check
```

### Étape 1.3: Test individuel
- [ ] Test SAM3 seul (script minimal)
- [ ] Test DA3 seul (script minimal)
- [ ] Test les deux ensemble (même script)

**Livrable**: Script `test_compatibility.py` qui charge SAM3 et DA3 sans erreur

---

## Phase 2: Wrappers Simples
**Objectif**: Code Python simple qui wrap SAM3 et DA3

### Étape 2.1: SAM3 Wrapper Minimal
```python
# sam3_simple.py
import torch
from sam3 import ...

class SAM3Simple:
    def __init__(self):
        self.model = None

    def load(self):
        # Charger SAM3
        pass

    def segment_points(self, image, points, labels):
        # Segmenter avec points
        return mask
```

**Test**: `python test_sam3.py image.jpg` → génère mask.png

### Étape 2.2: DA3 Wrapper Minimal
```python
# depth_simple.py
import torch
from depth_anything_3 import ...

class DA3Simple:
    def __init__(self):
        self.model = None

    def load(self):
        # Charger DA3
        pass

    def estimate_depth(self, image):
        # Estimer depth
        return depth_map
```

**Test**: `python test_da3.py image.jpg` → génère depth.png

### Étape 2.3: Test Combiné
```python
# test_both.py
from sam3_simple import SAM3Simple
from depth_simple import DA3Simple

sam = SAM3Simple()
sam.load()

depth = DA3Simple()
depth.load()

# Tester sur même image
# Vérifier mémoire GPU
```

**Livrable**: Les deux modèles chargent sans conflit de mémoire

---

## Phase 3: Integration Simple
**Objectif**: Un script qui utilise les deux

### Étape 3.1: Pipeline Basique
```python
# pipeline.py
class SimplePipeline:
    def __init__(self):
        self.sam3 = SAM3Simple()
        self.depth = DA3Simple()

    def load_models(self):
        self.sam3.load()
        self.depth.load()

    def process_image(self, image_path, points):
        # 1. Segmentation
        mask = self.sam3.segment_points(image, points, labels)

        # 2. Depth
        depth = self.depth.estimate_depth(image)

        return mask, depth
```

**Test**: Script CLI simple
```bash
python pipeline.py image.jpg --points 100,100 200,200
```

**Livrable**: CLI qui produit mask + depth pour une image

---

## Phase 4: GUI Minimal (optionnel)
**Objectif**: Interface graphique basique

### Étape 4.1: GUI Tkinter Simple
- Charger image
- Cliquer pour ajouter points
- Bouton "Segment"
- Bouton "Depth"
- Afficher résultats

**Livrable**: GUI fonctionnelle basique

---

## Phase 5: Packaging
**Objectif**: Installation propre

### Étape 5.1: Structure Finale
```
ultimate-rotoscopy/
├── pyproject.toml
├── requirements.txt
├── install.sh
├── src/
│   └── ultimate_roto/
│       ├── __init__.py
│       ├── sam3_wrapper.py
│       ├── depth_wrapper.py
│       └── pipeline.py
├── tests/
│   ├── test_sam3.py
│   ├── test_depth.py
│   └── test_pipeline.py
└── examples/
    └── demo.py
```

**Livrable**: Package installable avec `pip install -e .`

---

## Notes Importantes

### Priorités
1. **SIMPLICITÉ** > Fonctionnalités
2. **ÇA MARCHE** > Architecture parfaite
3. **PAS DE FALLBACKS** - SAM3 et DA3 uniquement
4. **TESTS À CHAQUE ÉTAPE** - Ne pas avancer si ça crash

### Anti-patterns à éviter
- ❌ Pas de mélange v2/v3
- ❌ Pas de fallbacks automatiques
- ❌ Pas de wrapper sur wrapper
- ❌ Pas d'architecture complexe avant que ça marche
- ❌ Pas de threading avant d'avoir testé synchrone

### Checklist de validation
Chaque phase doit passer:
- [ ] Importe sans erreur
- [ ] Charge les modèles sans crash
- [ ] Traite une image de test
- [ ] Pas de memory leak
- [ ] Peut être appelé plusieurs fois

---

## Prochaine Action

**PHASE 1 - ÉTAPE 1.1**: Analyser les requirements réels de SAM3 et DA3
```bash
# Cloner les repos pour lire requirements
git clone https://github.com/facebookresearch/sam3
git clone https://github.com/ByteDance-Seed/Depth-Anything-3

# Lire setup.py / pyproject.toml / requirements.txt
# Identifier versions exactes compatibles
```

**Question**: Voulez-vous que je commence par analyser les requirements des deux repos pour trouver la configuration compatible?
