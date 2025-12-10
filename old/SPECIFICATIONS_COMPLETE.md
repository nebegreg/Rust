# Ultimate Rotoscopy - Cahier des Charges Complet
# Application VFX Professionnelle

## 🎯 Vision

**Application de rotoscopy de niveau production** pour artistes VFX, combinant les dernières innovations AI (SAM3, Depth Anything V3, MatAnyone) avec un workflow professionnel compatible avec les pipelines existants (Flame, Nuke, etc.).

---

## 👥 Utilisateurs Cibles

- **Artistes rotoscopy** dans studios VFX
- **Compositeurs** nécessitant des mattes précis
- **Superviseurs VFX** gérant des shots complexes
- **Motion designers** pour effets créatifs

**Niveau**: Professionnel (pas grand public)

---

## 🎬 Use Cases Principaux

### Use Case 1: Extraction de Personnage pour Compositing
```
Input: Plan avec acteur sur green screen imparfait
Workflow:
  1. Importer séquence vidéo (24-60 fps, 2K-4K)
  2. Sélectionner personnage avec points SAM3 (frame 1)
  3. Propagation temporelle auto sur toute la séquence
  4. Raffiner alpha matte avec MatAnyone (cheveux, bords)
  5. Export EXR séquence vers Flame
Output: Alpha matte propre, frame-accurate, production-ready
```

### Use Case 2: Isolation d'Objet avec Depth
```
Input: Shot avec plusieurs éléments à séparer
Workflow:
  1. Segmentation multi-objets (SAM3)
  2. Génération depth map (DA3) pour Z-depth
  3. Séparation foreground/midground/background
  4. Mattes individuels par élément
  5. Export multi-layer EXR avec AOVs
Output: Mattes + depth + normals pour relighting
```

### Use Case 3: Rotoscopy Batch sur Séquence
```
Input: 500 frames nécessitant le même traitement
Workflow:
  1. Définir région d'intérêt (ROI)
  2. Auto-tracking SAM3 sur séquence complète
  3. Corrections manuelles sur keyframes
  4. Interpolation/propagation automatique
  5. Batch export
Output: Séquence complète traitée en production
```

---

## 🔧 Fonctionnalités Essentielles

### 1. Gestion Vidéo/Séquences

#### Import
- [x] Formats: MP4, MOV, AVI, WebM
- [x] Séquences images: EXR, DPX, PNG, TIFF, JPG
- [x] Résolutions: SD → 8K
- [x] Frame rates: 23.976 → 120 fps
- [x] Color spaces: sRGB, Rec709, ACES, Linear

#### Lecture
- [x] Timeline scrubbing professionnel
- [x] Playback temps réel (ou proche)
- [x] In/Out points
- [x] Frame-accurate navigation
- [x] Thumbnails pour navigation rapide

### 2. Segmentation (SAM3)

#### Prompts Interactifs
- [x] **Points**: Foreground/background clicks
- [x] **Box**: Rectangle de sélection
- [x] **Text** (SAM3 feature): "person wearing red jacket"
- [x] **Visual exemplar**: Référence d'un autre frame

#### Propagation Temporelle
- [x] Forward/backward tracking automatique
- [x] Détection d'occlusions
- [x] Keyframe system pour corrections
- [x] Interpolation intelligente entre keyframes
- [x] Consistency temporelle (pas de flickering)

#### Options Avancées
- [x] Multi-object segmentation
- [x] Instance separation (personne A vs B)
- [x] Edge refinement toggle
- [x] Mask dilation/erosion
- [x] Feathering controls

### 3. Alpha Matting (MatAnyone)

#### Raffinement de Matte
- [x] Detail preservation (cheveux, fourrure, verre)
- [x] Edge smoothing professionnel
- [x] Transparency handling
- [x] Motion blur consideration
- [x] Temporal coherence sur séquence

#### Contrôles Artiste
- [x] Core matte (opaque certain)
- [x] Edge matte (transition)
- [x] Manual touch-up tools
- [x] Before/after comparison
- [x] Matte quality metrics (coverage, edge quality)

### 4. Depth & 3D (Depth Anything V3)

#### Estimation de Profondeur
- [x] Depth map relative
- [x] Depth map metric (si possible)
- [x] Normal maps pour relighting
- [x] Per-frame ou sequence-consistent

#### Utilisation
- [x] Z-depth AOV export
- [x] Depth-based masking (near/far planes)
- [x] 3D camera estimation (DA3 feature)
- [x] Point cloud visualization
- [x] Integration avec depth de compositing

### 5. Export Professionnel

#### Formats Flame
```python
# Flame GMask Tracer format
- .gmask files
- Node structure compatible
- Metadata preservation
```

#### Formats Nuke
```python
# Nuke-compatible outputs
- .nk scripts avec nodes
- Roto/RotoPaint data
- Cryptomatte support
```

#### Formats Standards
```
EXR Sequences:
  - Multi-layer: beauty, matte, depth, normals
  - 16-bit float / 32-bit float
  - Compression: ZIP, PIZ, DWAA
  - Metadata: timecode, frame info, render stats

DPX Sequences:
  - 10-bit / 16-bit
  - Log color space
  - Cinema-grade

Image Sequences:
  - PNG (8/16-bit)
  - TIFF (8/16/32-bit)
  - Numbered: shot_####.ext
```

#### AOVs (Arbitrary Output Variables)
```
Channels d'export:
  - RGBA (beauty)
  - matte.R (alpha matte)
  - depth.Z (depth map)
  - N.XYZ (normals)
  - id.instanceID (object IDs)
  - motion.XY (motion vectors)
```

### 6. Interface Utilisateur

#### Layout Professionnel
```
+------------------+------------------------+------------------+
|   Timeline       |      Viewport          |   Inspector      |
|   (playback)     |  (canvas + tools)      |   (properties)   |
+------------------+------------------------+------------------+
|   Thumbnails     |   Tool Palette         |   Layers         |
+------------------+------------------------+------------------+
```

#### Workspace Areas
1. **Viewport**: Canvas avec overlay matte/depth
2. **Timeline**: Scrubbing, keyframes, cache status
3. **Layers**: Mattes multiples, blend modes
4. **Properties**: Paramètres modèles, refinement
5. **Tools**: Brush, eraser, selection tools

#### Keyboard Shortcuts (Standard VFX)
```
Playback:
  Space: Play/Pause
  J/K/L: Rewind/Stop/Forward
  I/O: Set In/Out points
  [ / ]: Previous/Next keyframe

Tools:
  Q: Selection
  W: Move
  B: Brush
  E: Eraser
  R: Rotation
  S: Scale

View:
  F: Frame selected
  A: Show all
  ~/`: Toggle overlay
  Alt+Drag: Pan
  Scroll: Zoom
```

### 7. Performance & Optimization

#### Requirements
- [x] RTX 4090 optimization
- [x] CUDA 12.6 support
- [x] Multi-GPU support (optional)
- [x] Background processing
- [x] Smart caching system

#### Caching Strategy
```python
Cache Levels:
  1. Image cache (decoded frames)
  2. Embedding cache (SAM3 image encodings)
  3. Mask cache (computed masks)
  4. Matte cache (refined alphas)
  5. Depth cache (depth maps)

Memory Management:
  - LRU eviction
  - Disk spill for large sequences
  - Configurable cache size
  - Cache warming (preload)
```

### 8. Workflow Integration

#### Autodesk Flame
- GMask Tracer export
- Action node integration
- Batch node compatibility
- Metadata preservation

#### Foundry Nuke
- .nk script generation
- Roto node export
- Smart vector integration
- Cryptomatte support

#### DaVinci Resolve
- Fusion page compatibility
- Color page masks
- Power windows export

#### After Effects
- Shape layer export
- Mask path data
- Plugin integration

---

## 🏗️ Architecture Technique

### Stack Technologique

#### Backend (Python)
```
Core ML:
  - SAM3 (segmentation + tracking)
  - Depth Anything V3 (depth estimation)
  - MatAnyone (alpha matting)

Video Processing:
  - OpenCV (decoding/encoding)
  - MoviePy (video manipulation)
  - imageio (formats support)
  - PyAV (advanced codecs)

Numerical:
  - NumPy (array operations)
  - SciPy (signal processing)
  - PyTorch (GPU acceleration)

Export:
  - OpenEXR (EXR writing)
  - Pillow (image formats)
  - Custom Flame/Nuke exporters
```

#### Frontend (GUI)
```
Framework: PySide6 (Qt)
  - Professional dock system
  - OpenGL viewport
  - Timeline widget custom
  - HDR display support

Visualization:
  - Matplotlib (graphs/metrics)
  - PyQtGraph (real-time plots)
  - OpenGL shaders (overlays)
```

#### Architecture Patterns
```python
Model-View-Controller:
  - Models: SAM3, DA3, MatAnyone wrappers
  - Views: GUI components, viewport
  - Controllers: Processing pipeline, state management

Observer Pattern:
  - Progress updates
  - Frame cache notifications
  - Processing completion events

Command Pattern:
  - Undo/Redo system
  - Macro recording
  - Batch operations
```

---

## 📊 Spécifications Techniques

### Performance Targets (RTX 4090)

```
SAM3 Segmentation:
  - Loading: < 10s
  - First frame: < 2s
  - Subsequent: < 0.5s (cached encoding)
  - 4K frame: < 1s

Depth Estimation:
  - Loading: < 8s
  - Per frame: < 0.5s
  - 4K frame: < 1s

MatAnyone Matting:
  - Loading: < 12s
  - Per frame: < 2s (high quality)
  - 4K frame: < 3s

Sequence Processing:
  - 100 frames @ 1080p: < 2 min
  - 1000 frames @ 1080p: < 15 min
  - Batch mode: near real-time playback
```

### Memory Management
```
GPU Memory (24GB RTX 4090):
  - SAM3 model: ~4GB
  - DA3 model: ~3GB
  - MatAnyone: ~5GB
  - Frame cache: ~8GB
  - Headroom: ~4GB

RAM Requirements:
  - Minimum: 32GB
  - Recommended: 64GB
  - Heavy sequences: 128GB
```

### File Size Estimates
```
4K Frame Outputs:
  - PNG matte: ~5MB
  - EXR beauty: ~20MB
  - EXR multi-layer: ~40MB
  - DPX: ~25MB

Sequence (1000 frames @ 4K):
  - PNG sequence: ~5GB
  - EXR multi-layer: ~40GB
  - With caches: +10-20GB
```

---

## 🎨 User Experience

### Workflow Typique (5 minutes pour shot simple)

```
Minute 0-1: Import & Setup
  - Importer séquence
  - Définir working resolution
  - Prévisualiser

Minute 1-2: Segmentation
  - Frame 1: Cliquer points SAM3
  - Vérifier mask qualité
  - Lancer propagation

Minute 2-3: Propagation & Corrections
  - Propagation auto sur séquence
  - Identifier frames problématiques
  - Corrections manuelles sur keyframes

Minute 3-4: Raffinement
  - MatAnyone sur frames clés
  - Vérifier edge quality
  - Ajustements finaux

Minute 4-5: Export
  - Choisir format (EXR multi-layer)
  - Définir AOVs nécessaires
  - Export vers Flame/Nuke
```

### Indicateurs de Succès

**Qualité**:
- Edge accuracy: < 0.5 pixel error
- Temporal stability: < 2% flicker
- Detail preservation: 95%+ (cheveux, etc.)

**Vitesse**:
- 10x plus rapide que rotoscopy manuel
- Corrections: < 20% du temps total
- Export: temps réel ou mieux

**Fiabilité**:
- Crash rate: < 0.1%
- Cache hit rate: > 90%
- Export success: 99.9%

---

## 🚀 Phases de Développement

### Phase 1: Core Foundation ✓
- [x] Environment setup (SAM3 + DA3)
- [x] Compatibility verified
- [ ] Basic video loading
- [ ] Single frame processing

### Phase 2: Segmentation Pipeline
- [ ] SAM3 integration complète
- [ ] Point/box/text prompts
- [ ] Multi-frame propagation
- [ ] Keyframe system
- [ ] Cache system

### Phase 3: Matting Integration
- [ ] MatAnyone integration
- [ ] Matte refinement pipeline
- [ ] Quality controls
- [ ] Temporal consistency

### Phase 4: Depth & 3D
- [ ] DA3 integration
- [ ] Normal map generation
- [ ] Depth-based operations
- [ ] 3D visualization

### Phase 5: Professional Export
- [ ] EXR multi-layer writer
- [ ] Flame GMask export
- [ ] Nuke script generation
- [ ] Metadata system
- [ ] AOV management

### Phase 6: GUI Professionnel
- [ ] Timeline avec scrubbing
- [ ] Viewport avec overlays
- [ ] Tool palette
- [ ] Inspector/Properties
- [ ] Keyboard shortcuts

### Phase 7: Optimization
- [ ] GPU optimization
- [ ] Multi-threading
- [ ] Cache tuning
- [ ] Batch processing

### Phase 8: Production Ready
- [ ] Testing sur vrais shots
- [ ] Documentation complète
- [ ] Training materials
- [ ] Support pipeline intégration

---

## 📝 Questions pour Affiner

1. **Format prioritaire**: Flame ou Nuke? (ou les deux?)
2. **Type de shots**: Principalement greenscreen, ou tous types?
3. **Résolution habituelle**: 2K, 4K, ou mixte?
4. **Longueur séquences**: Quelques frames ou milliers?
5. **Features prioritaires**: Vitesse ou qualité ultime?

---

**Prochaine étape**: Valider ce cahier des charges et commencer l'architecture détaillée.

Qu'en pensez-vous? Manque-t-il des features critiques pour votre usage?
