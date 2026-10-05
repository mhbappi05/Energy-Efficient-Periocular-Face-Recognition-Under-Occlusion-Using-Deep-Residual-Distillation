# Energy-Efficient Periocular Face Recognition Under Occlusion Using Deep Residual Distillation

Research project (Digital Image Processing) that recognizes people from the **periocular region** (eyes and eyebrows) when the lower face is occluded (e.g. masks), and compresses a heavy CNN into a lighter model with **knowledge distillation** for energy-efficient inference.

Originally developed in **Google Colab**; this repository packages the notebooks, processed data layout, trained weights, and result figures for GitHub.

---

## Motivation

Standard face recognition often fails when the mouth and nose are covered. The periocular region remains visible and identity-discriminative. Large models such as ResNet-50 work well but are costly in disk space, FLOPs, and latency. This work:

1. Crops periocular regions with facial landmarks  
2. Trains a ResNet-50 **teacher** for identity classification  
3. Distills knowledge into a ResNet-18 **student** for a smaller, faster model  
4. Reports accuracy **and** green/efficiency metrics (size, latency, FLOPs)

---

## Method Overview

```
Face images (often masked / occluded)
        │
        ▼
 dlib HOG face detector + 68-point landmarks
        │
        ▼
 Periocular crop (landmarks 17–47, +20 px padding)
        │
        ▼
 Train / test split (80 / 20 per identity)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 ResNet-50 Teacher              ResNet-18 Student
 (cross-entropy)               (hard CE + soft KL distillation)
        │                              │
        └────────── metrics & figures ─┘
```

**Distillation loss** (local phase):

\[
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}} + (1-\alpha) \cdot T^{2} \cdot \mathrm{KL}\big(\sigma(z_s/T) \,\|\, \sigma(z_t/T)\big)
\]

with temperature \(T = 3.0\) and \(\alpha = 0.5\).

---

## Project Phases

| Phase | Notebook | Dataset | Classes | Goal |
|-------|----------|---------|---------|------|
| **Global** | `DIP.ipynb` | AFDB-derived periocular crops | **469** identities | Train ResNet-50 teacher; evaluate; efficiency baselines |
| **Local** | `localData.ipynb` | Custom local face set | **18** people | Transfer teacher features; distill ResNet-18 student |

### Global pipeline (`DIP.ipynb`)

1. Install `dlib` / OpenCV; load 68-landmark predictor  
2. Detect faces and extract periocular crops from `AFDB_face_dataset`  
3. Save crops to `Processed_Periocular/`  
4. Split into `Model_Data/train` and `Model_Data/test`  
5. Fine-tune ImageNet-pretrained **ResNet-50** (112×112 inputs, Adam, augmentation)  
6. Checkpoint to `teacher_checkpoint.pth`; track FLOPs / time  
7. Evaluate test metrics and plot efficiency / confusion / F1 figures  

### Local pipeline (`localData.ipynb`)

1. Same periocular extraction on `Local Dataset/` → `Local_Processed_Periocular/`  
2. Split into `Local_Model_Data/train` and `Local_Model_Data/test`  
3. Transfer ResNet-50 backbone weights (drop FC); fine-tune for 18 classes  
4. Distill into **ResNet-18** student; save `local_student_periocular.pth`  
5. Classification report, confusion matrix, F1 bars, loss curves  

---

## Repository Structure

```
.
├── DIP.ipynb                      # Global (AFDB / 469-class) Colab notebook
├── localData.ipynb                # Local (18-class) distillation notebook
├── AFDB_face_dataset/             # Raw global face images (by identity)
├── Processed_Periocular/          # Periocular crops (global)
├── Model_Data/                    # train/ + test/ ImageFolder splits (global)
├── Model_Data.zip                 # Packed Model_Data for faster Colab I/O
├── Local Dataset/                 # Raw local faces (18 subjects)
├── Local_Processed_Periocular/    # Periocular crops (local)
├── Local_Model_Data/              # train/ + test/ splits (local)
├── teacher_checkpoint.pth         # ResNet-50 teacher weights (~280 MB)
├── student_final_distilled.pth    # Distilled student (global-scale artifact)
├── local_student_periocular.pth   # Distilled ResNet-18 for 18 local IDs
└── DIP Research Grpah images/     # Result plots
    ├── GlobalData/
    └── LocalData/
```

---

## Datasets

### Global — AFDB-based periocular set

- Source images under `AFDB_face_dataset/` (masked / face images organized by identity)  
- After landmark cropping: `Processed_Periocular/` (~727 subject folders; **469** identities with enough images enter the train/test split)  
- Modeling layout: `Model_Data/train` and `Model_Data/test` (ImageFolder)

### Local — custom 18-subject set

Subjects: `ahlow`, `ahmad`, `anik`, `asif`, `dipto`, `ezaz`, `hemal`, `maisha`, `masum`, `milion`, `omar`, `pulok`, `saikat`, `sakib`, `shimul`, `tanjim`, `zobayer`, `zonayed`

- Raw: `Local Dataset/`  
- Crops: `Local_Processed_Periocular/`  
- Splits: `Local_Model_Data/train`, `Local_Model_Data/test`

---

## Models & Checkpoints

| File | Architecture | Role |
|------|--------------|------|
| `teacher_checkpoint.pth` | ResNet-50 | Global teacher (dict with `model_state_dict`, optimizer, epoch, accuracy) |
| `student_final_distilled.pth` | ResNet-18 (distilled) | Lightweight student artifact |
| `local_student_periocular.pth` | ResNet-18 | Distilled student for 18 local identities |

**Typical training settings**

- Input size: **112 × 112**  
- Normalize: ImageNet mean/std `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`  
- Teacher: Adam, data augmentation (flip / rotation / brightness)  
- Student distillation: \(T=3\), \(\alpha=0.5\), ~15 epochs (local)

---

## Key Results

### Global teacher (469 classes)

| Metric | Value |
|--------|------:|
| Overall test accuracy | **71.21%** |
| Macro F1-score | **68.90%** |
| Weighted F1-score | **73.63%** |

### Local distilled student (18 classes)

| Metric | Value |
|--------|------:|
| Overall accuracy | **93.10%** |
| Macro-average F1 | **85.34%** |

### Efficiency (teacher vs student)

| Metric | Teacher (ResNet-50) | Student (ResNet-18) |
|--------|--------------------:|--------------------:|
| Disk size | ~280 MB | ~43.6 MB |
| Inference | ~7.40 ms / image | ~2.92 ms / image |
| Compute | ~125,630 GFLOPs* | ~28,000 GFLOPs* |

\*As reported in the project efficiency figures / notebook comparisons.

Result figures live under `DIP Research Grpah images/GlobalData/` and `.../LocalData/` (loss curves, confusion matrices, F1 charts, hardware footprints).

---

## Setup & How to Run

### Option A — Google Colab (original workflow)

1. Upload this repo (or mount Drive with the same folder layout).  
2. Open `DIP.ipynb` or `localData.ipynb`.  
3. Update Drive paths if needed (cells use `/content/drive/MyDrive/DIP/...`).  
4. Run setup cells (`dlib`, landmark `.dat` download, Drive mount).  
5. Run preprocessing → split → train → evaluate → plots.

For faster training I/O on Colab, unzip `Model_Data.zip` onto local Colab disk (as in the notebook) instead of reading every image from Drive.

### Option B — Local / Jupyter

```bash
# Python 3.10+ recommended
pip install torch torchvision opencv-python-headless dlib scikit-learn matplotlib seaborn pandas scipy
```

Download the dlib shape predictor:

```text
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

Point notebook paths to this repository root instead of `/content/drive/MyDrive/DIP/...`, then run cells in order.

**Note:** Cells that call `google.colab` (`drive.mount`, `cv2_imshow`) are Colab-specific; replace them with local paths / `cv2.imshow` or `matplotlib` when running offline.

---

## Dependencies

- Python 3.x  
- PyTorch + torchvision  
- OpenCV (`opencv-python` / `opencv-python-headless`)  
- dlib (+ `shape_predictor_68_face_landmarks.dat`)  
- scikit-learn, NumPy, Matplotlib, Seaborn, Pandas, SciPy  

GPU (CUDA) is recommended for training; CPU works for small local experiments.

---

## GitHub notes (large files)

| Tracked how | What |
|--------------|------|
| **Normal git** | Notebooks, `README.md`, result figures (`DIP Research Grpah images/`) |
| **Git LFS** (`*.pth`, `*.zip`) | Model checkpoints + `Model_Data.zip` |
| **Ignored** (`.gitignore`) | Image datasets: `AFDB_face_dataset/`, `Local Dataset/`, `Processed_Periocular/`, `Local_Processed_Periocular/`, `Model_Data/`, `Local_Model_Data/` |

**One-time setup** (if cloning or contributing):

```bash
git lfs install
git lfs pull   # after clone, to download weights / zip
```

Approximate LFS payload: ~492 MB (`teacher_checkpoint.pth` ~280 MB + two student `.pth` files + `Model_Data.zip`). Free GitHub LFS includes **1 GB** storage — enough for checkpoints, not for full image trees. Keep datasets on Drive or regenerate from the notebooks.

Download the dlib landmark file yourself (not in the repo):

```text
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

---

## Citation / Course context

Digital Image Processing research project — periocular recognition under occlusion with deep residual knowledge distillation for energy-efficient deployment.

---

## License

Add a license of your choice (e.g. MIT) if you plan to open the repository publicly. Dataset redistribution may be subject to the original AFDB / source dataset terms.
