# BUGS CRITIQUES À CORRIGER

## ✅ CORRIGÉS

### 1. ✅ SAM3 Import Error (BLOQUANT)
**Problème**: Import de `SAM3` échoue car seul `SAM3Segmentor` existe
**Fix**: Ajout d'alias `SAM3 = SAM3Segmentor` dans `sam3.py:1243`
**Status**: ✅ CORRIGÉ

### 2. ✅ Versions Incohérentes
**Problème**: pyproject.toml (1.0.0) vs __init__.py (3.0.0) vs cli (3.0.0)
**Fix**: Alignement à 1.0.0 partout
**Status**: ✅ CORRIGÉ

### 3. ✅ Gradio GUI - Points Effacés Avant Utilisation
**Problème**: `process_image()` fait `points.clear()` puis `if len(points) > 0`
**Fix**: Suppression de `points.clear()` - utilisation du bouton dédié
**Fichier**: `src/ultimate_rotoscopy/gui.py:75-76`
**Status**: ✅ CORRIGÉ

---

## ❌ À CORRIGER (Priorité Haute)

### 4. ❌ Rust - Point Cloud Normals Incorrects (ALGORITHME FAUX)
**Fichier**: `src/lib.rs:716-732`
**Problème**: Commentaire dit "smallest eigenvector" mais power iteration converge vers le PLUS GRAND
**Impact**: Normales de point cloud complètement fausses

**Solution Recommandée**:
```rust
// Utiliser nalgebra::SymmetricEigen au lieu de power iteration
use nalgebra::{Matrix3, SymmetricEigen};

let cov_matrix = Matrix3::new(
    cov[0][0], cov[0][1], cov[0][2],
    cov[1][0], cov[1][1], cov[1][2],
    cov[2][0], cov[2][1], cov[2][2],
);

let eigen = SymmetricEigen::new(cov_matrix);
// Trouver l'index du plus petit eigenvalue
let min_idx = eigen.eigenvalues.argmin().0;
let normal_vector = eigen.eigenvectors.column(min_idx);
```

### 5. ❌ Rust - Division par Zéro dans `alpha::feather_alpha`
**Fichier**: `src/lib.rs` (module alpha)
**Problème**: Si `radius == 0.0` → division par zéro dans `dist / radius`
**Fix**:
```rust
pub fn feather_alpha(alpha: ..., radius: f32) -> ... {
    if radius <= 0.0 {
        return alpha.to_owned(); // Retourner alpha direct
    }
    // ... reste du code
}
```

### 6. ❌ Rust - Division par Zéro dans `depth::bilateral_filter`
**Fichier**: `src/lib.rs` (module depth)
**Problème**: Si `spatial_sigma == 0` ou `range_sigma == 0` → NaN
**Fix**:
```rust
pub fn bilateral_filter(depth: ..., spatial_sigma: f32, range_sigma: f32) -> ... {
    let spatial_sigma = spatial_sigma.max(1e-6);
    let range_sigma = range_sigma.max(1e-6);
    // ... reste du code
}
```

### 7. ❌ Rust - Division par Zéro dans `edge::refine_edges`
**Fichier**: `src/lib.rs` (module edge)
**Problème**: `epsilon` utilisé comme diviseur dans poids gaussien
**Fix**:
```rust
pub fn refine_edges(..., epsilon: f32) -> ... {
    let epsilon = epsilon.max(1e-6);
    // ... reste du code
}
```

### 8. ❌ Rust - `edge::blur` Retourne Image Noire pour Petites Images
**Fichier**: `src/lib.rs` (module edge)
**Problème**:
- Initialise `result` à zéros partout
- Boucles: `y in 2..h-2` / `x in 2..w-2`
- Si image < 5x5 → boucles ne tournent pas → image noire

**Fix**:
```rust
pub fn blur(image: ArrayView2<f32>) -> Array2<f32> {
    let (h, w) = image.dim();

    // Si image trop petite, retourner copie
    if h < 5 || w < 5 {
        return image.to_owned();
    }

    // Ou initialiser result = image.to_owned() au lieu de zeros
    let mut result = image.to_owned();

    // Puis flouter seulement l'intérieur
    for y in 2..h-2 {
        for x in 2..w-2 {
            // ... blur logic
        }
    }
    result
}
```

---

## ❌ À CORRIGER (Priorité Moyenne)

### 9. ❌ Pipeline - Paramètre `mask` Ignoré
**Fichier**: `src/ultimate_rotoscopy/pipeline/unified.py`
**Problème**: `process_image(..., mask=...)` accepte mask mais ne l'utilise JAMAIS
**Fix**: Utiliser mask comme matte d'entrée si fourni

### 10. ❌ Pipeline - Double Processing dans `process_batch()`
**Fichier**: `src/ultimate_rotoscopy/pipeline/unified.py`
**Problème**:
- Traite image via `process_image()`
- Puis retraite si prompt fourni
- Ne sauve pas le résultat → coût x2

**Fix**: Passer prompt directement à `process_image()` ou sauver résultat

### 11. ❌ Exceptions Silencieuses (`except: pass`)
**Fichiers**:
- `src/ultimate_rotoscopy/acceleration/caching.py` - `prefetch()`
- `src/ultimate_rotoscopy/acceleration/multi_gpu.py` - `update_memory_info()`
- `src/ultimate_rotoscopy/gui/backend.py` - plusieurs endroits

**Problème**: Erreurs avalées → debug impossible
**Fix**: Au minimum logger:
```python
except Exception as e:
    logger.exception(f"Error in {function_name}: {e}")
```

### 12. ❌ Dépendances Conflictuelles
**Fichier**: `requirements.txt`
**Problèmes**:
- `opencv-python` ET `opencv-python-headless` (mutuellement exclusifs)
- `onnxruntime` ET `onnxruntime-gpu` (selon plateforme)

**Fix**: Choisir une variante selon contexte:
```txt
# Pour GUI
opencv-python>=4.8.0

# Pour serveur headless
# opencv-python-headless>=4.8.0

# GPU (Linux/Windows avec CUDA)
# onnxruntime-gpu>=1.16.0

# CPU ou macOS
onnxruntime>=1.16.0
```

---

## 📊 Statistiques

**Total Bugs**: 12
**Corrigés**: 3 (25%)
**Critiques Restants**: 5 (Rust)
**Moyens Restants**: 4 (Python)

---

## 🚨 Impact

**BLOQUANTS** (empêchent démarrage):
- ✅ SAM3 import → CORRIGÉ

**CRITIQUES** (résultats incorrects):
- ❌ Point cloud normals → Normales fausses
- ❌ Divisions par zéro → Crash/NaN

**IMPORTANTS** (bugs logiques):
- ❌ Gradio points → Segmentation impossible (CORRIGÉ)
- ❌ Pipeline mask → Fonctionnalité manquante
- ❌ Exceptions silencieuses → Debug impossible

---

## 📋 Ordre de Correction Recommandé

1. ✅ SAM3 alias (FAIT)
2. ✅ Versions (FAIT)
3. ✅ Gradio points (FAIT)
4. ❌ **Rust normales** (URGENT - résultats faux)
5. ❌ **Rust divisions/0** (URGENT - crashes)
6. ❌ Pipeline mask
7. ❌ Exceptions logging
8. ❌ Dépendances

---

## ✅ Pour Tester les Fixes

```bash
# Test import SAM3
cd src && python3 -c "from ultimate_rotoscopy.models.sam3 import SAM3, SAM3Segmentor; assert SAM3 is SAM3Segmentor; print('✓ SAM3 OK')"

# Test versions
python3 -c "from ultimate_rotoscopy import __version__; assert __version__ == '1.0.0'; print('✓ Version OK')"

# Rust compilation
cargo build --release

# Tests Rust
cargo test
```

---

**Créé**: 2025-12-10
**Source**: Analyse ChatGPT + Correction Claude
