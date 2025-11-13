# ELEC5305 — UrbanSound8K Automobile Sound Recognition

**Author:** Tushar Manish Khupte (SID: 520330504)  
**Unit:** ELEC5305 — Audio Processing & DSP  
**The University of Sydney**

[![Open In Colab](https://img.shields.io/badge/Open%20In%20Colab-Deep%20Learning%20Notebook-orange?logo=googlecolab)](https://colab.research.google.com/drive/1tQ3kxnaScF5GZLx7VYJQ2cxjm3pbHZN9?usp=sharing)

> Reproducible end-to-end **UrbanSound8K** pipeline with a focus on **automobile-related sounds**  
> (e.g. car horn, engine idling, siren), including training code, metrics, plots, and audio examples.

---

## 🎯 Project Overview

This repository contains my **individual ELEC5305 final project**. The goal is to recognise non-speech acoustic events in urban environments with a focus on **automobile-related hazards**, using the **UrbanSound8K** dataset.

The project:

- Implements a full **data → features → model → evaluation → examples** pipeline.
- Compares a **CNN baseline** against a **hybrid CNN–BiLSTM model** with auxiliary features.
- Uses **macro-F1** and **per-class F1** to highlight performance across classes, not just overall accuracy.
- Exports **plots, confusion matrix, logs, metrics, and audio pairs** for qualitative analysis.

This repo is designed so that markers can **see and hear** what the model is doing, not just read numbers.

---

## ✨ Key Features

- **UrbanSound8K automation**
  - Assumes the standard UrbanSound8K folder layout.
  - Uses caching to avoid recomputing features on each run.

- **Rich feature set**
  - Log-mel spectrograms
  - Zero-Crossing Rate (ZCR)
  - Root Mean Square (RMS) energy
  - Modulation spectrum summary
  - Linear Predictive Coding (LPC) coefficients

- **Models**
  - Baseline: compact **CNN** (e.g. `TinyCNNWithAux`-style).
  - Proposed: **CNN → BiLSTM** temporal head + auxiliary feature branch.

- **Training strategy**
  - Clear **train / val / test** split by folds.
  - **Early stopping on validation macro-F1**.
  - Checkpointing of the **best** model.

- **Outputs & explainability**
  - Confusion matrix, training curves, per-class F1 plots.
  - Example WAV files for:
    - Raw input
    - Denoised / processed versions
    - Feature-space (mel) reconstructions
  - Audio pairs showing `Input → Denoised → Mel-recon` for the same clip.

---

## 📁 Repository Structure

```text
content/
 ├─ UrbanSound8K/               # (Expected) dataset root (not included; see below)
 ├─ outputs/                    # All generated artifacts
 │   ├─ training_log.csv        # Epoch-wise metrics
 │   ├─ metrics.json            # Final metrics & per-class scores
 │   ├─ confusion_matrix.png    # Final confusion matrix
 │   ├─ plot_accuracy.png       # Training/validation accuracy vs epochs
 │   ├─ plot_loss.png           # Training/validation loss vs epochs
 │   ├─ plot_macro_f1.png       # Macro-F1 vs epochs
 │   ├─ plot_per_class_f1.png   # Per-class F1 scores
 │   ├─ examples/
 │   │   ├─ audio_pairs/        # Input → Denoised → Mel-reconstruction audio triplets
 │   │   ├─ test_mistakes_audio/        # Misclassified examples (for error analysis)
 │   │   ├─ Feature-Space_Reconstruction/  # Audio reconstructed from mel features
 │   │   └─ ...                  # Other example subfolders
 │   └─ ...
 └─ us8k_cache/                 # Cached features to speed up future runs

ultrasound_8k_baseline_final_project.py   # Main training / evaluation / export script

outputs.zip                               # Zipped copy of the outputs/ directory (for markers)
.gitattributes
.gitignore
README.md
desktop.ini
````

> **Note:** `outputs.zip` is included so that markers can inspect all artifacts (plots, logs, audio) without needing to rerun training.

---

## 🧰 Requirements

Tested with:

* Python 3.10+ (recommended)
* Typical scientific audio stack:

  * `torch`, `torchaudio`
  * `librosa`
  * `numpy`, `pandas`
  * `scikit-learn`
  * `matplotlib`
  * `tqdm`

You can install these into a fresh virtual environment with:

```bash
pip install torch torchaudio librosa numpy pandas scikit-learn matplotlib tqdm
```

(Or adapt to your own environment / CUDA setup.)

---

## 🎵 UrbanSound8K Dataset

The project uses **UrbanSound8K**, a 10-class environmental sound dataset (Salamon et al., 2014).
You must obtain the dataset yourself (it is **not** bundled here due to size and licensing).

1. Download **UrbanSound8K** from the official source (or a mirrored academic repository).
2. Place the extracted folder so that the structure looks like:

```text
content/
 └─ UrbanSound8K/
     ├─ audio/
     ├─ metadata/
     └─ ...
```

If your dataset lives somewhere else, adjust the path in the script or configuration section accordingly.

> The code focuses on **automobile-related classes** (e.g. car horn, engine idling, siren), and may include a few confusable non-automobile classes depending on your configuration.

---

## 🚀 How to Run (Local)

Clone the repo:

```bash
git clone https://github.com/Tusharbhai07/elec5305-project-520330504.git
cd elec5305-project-520330504
```

Ensure your environment and dataset are set up (see sections above), then:

### 1. Train (and evaluate) the model

```bash
python ultrasound_8k_baseline_final_project.py
```

Typical behaviour:

* Loads or computes features (caching to `content/us8k_cache/`).
* Splits folds into **train / validation / test**.
* Trains the configured model with early stopping on **validation macro-F1**.
* Saves:

  * `content/outputs/best.pt` — best model checkpoint.
  * `content/outputs/training_log.csv`
  * `content/outputs/metrics.json`
  * Plots (`plot_*.png`) and `confusion_matrix.png`.

### 2. Regenerate plots only

If the script supports a plotting-only mode (no retraining), you can re-generate plots from existing logs:

```bash
python ultrasound_8k_baseline_final_project.py --plot
```

### 3. Export audio pairs / examples

To export matched audio examples (input → processed → reconstruction) and misclassified clips, use the provided audio export option:

```bash
python ultrasound_8k_baseline_final_project.py --audio 2
```

This will populate subfolders under:

* `content/outputs/examples/audio_pairs/`
* `content/outputs/examples/test_mistakes_audio/`
* `content/outputs/examples/Feature-Space_Reconstruction/`

*(Adjust the flag value as needed based on how you configured the script.)*

---

## 📊 Interpreting the Outputs

All key artifacts live in `content/outputs/`:

* **`training_log.csv`**
  Epoch-wise metrics (loss, accuracy, macro-F1, etc.).
  Can be loaded into Python, Excel, or plotted separately.

* **`metrics.json`**
  Final metrics summary for the best checkpoint, including:

  * Overall accuracy
  * Macro-F1
  * Per-class precision / recall / F1

* **Plots**

  * `plot_accuracy.png`
  * `plot_loss.png`
  * `plot_macro_f1.png`
  * `plot_per_class_f1.png`
  * `confusion_matrix.png`

* **Audio examples**

  * `examples/audio_pairs/`
    → For each selected test example, you can listen to:

    1. Raw input audio
    2. Denoised or processed version
    3. Mel-reconstructed version (what the model “hears”)

  * `examples/test_mistakes_audio/`
    → Misclassified clips for qualitative error analysis.

This combination of **numbers + plots + audio** supports both quantitative and qualitative evaluation in the ELEC5305 report.

---


## 🔗 Quick Links — Plots, Logs & Audio (click inside GitHub)

### 📈 Key Plots

- [Confusion Matrix](content/outputs/confusion_matrix.png)
- [Accuracy vs Epoch](content/outputs/plot_accuracy.png)
- [Loss vs Epoch](content/outputs/plot_loss.png)
- [Macro-F1 vs Epoch](content/outputs/plot_macro_f1.png)
- [Per-Class F1 Bar Plot](content/outputs/plot_per_class_f1.png)

### 📑 Metrics & Logs

- [Training Log (CSV)](content/outputs/training_log.csv)
- [Final Metrics (JSON)](content/outputs/metrics.json)
- [Best Model Checkpoint (`best.pt`)](content/outputs/best.pt)

### 🔊 Example Audio Folders

Browse these folders in GitHub and click any `.wav` file to listen:

- [Raw Input Examples](content/outputs/examples/some_examples_raw_Audio_INPUT/)
- [Denoised Output Examples](content/outputs/examples/some_examples_denoised_audio_OUTPUT/)
- [Mel-Reconstruction (Feature-Space)](content/outputs/examples/some_examples_Feature-Space_Reconstruction/)
- [Matched Audio Pairs (Input → Denoised → Mel-Recon)](content/outputs/examples/audio_pairs/)
- [Misclassified Test Clips](content/outputs/examples/test_mistakes_audio/)

### 🗃️ All Artifacts (ZIP)

If you just want everything in one go:

- [Download all outputs as ZIP](outputs.zip)

---
## 🧠 Model & Design Summary

### Baseline

* Input: log-mel spectrograms (with optional auxiliary descriptors).
* Architecture: small CNN-based classifier (`TinyCNNWithAux`-style).
* Training: supervised classification on UrbanSound8K folds.

### Proposed Model

* **Feature branch:**
  2D CNN stack over log-mel spectrograms (captures local time–frequency patterns).

* **Temporal branch:**
  Bi-directional LSTM (BiLSTM) to model frame-level temporal evolution.

* **Auxiliary branch:**
  Concatenates ZCR, RMS, modulation features, and LPC to the deep features before the classifier head.

* **Loss & selection:**
  Cross-entropy loss with early stopping based on **validation macro-F1**, not just accuracy, to avoid over-fitting to majority classes.

---

## 📌 Relation to the Baseline Code

This repository expands on a baseline CNN implementation (referred to as “Hassan MFCC+CNN” in the accompanying report) by:

1. **Changing input features**

   * MFCCs → **log-mel spectrograms** + auxiliary descriptors.

2. **Adding temporal modelling**

   * Pure CNN → **CNN + BiLSTM** hybrid.

3. **Data augmentation**

   * Integration of **SpecAugment-style** masks in the time/frequency domain (if enabled).

4. **Better model selection**

   * Switch from accuracy-based selection to **macro-F1**-based checkpointing.

5. **Explainability bundle**

   * Export of **audio pairs**, **mel reconstructions**, and **mistake galleries** for qualitative analysis.

The full and heavily commented training pipeline is also mirrored in the linked Colab notebook for easier inspection by markers.

---

## 🧪 Reproducing the Report Results

To reproduce the experiments used in the written ELEC5305 report:

1. Ensure **UrbanSound8K** is correctly placed under `content/UrbanSound8K/`.
2. Run the main training script once to generate:

   * `content/outputs/best.pt`
   * `content/outputs/metrics.json`
   * plots and logs.
3. (Optional) Run the script with `--audio` flags to generate example audio.
4. Base all tables, confusion matrices, and figures in the report directly on:

   * `metrics.json`
   * `confusion_matrix.png`
   * `plot_*.png`
   * exported audio examples.

This ensures the **report, README, and repository are fully consistent**.

---

## 📚 References (Short List)

1. J. Salamon, C. Jacoby, and J. P. Bello,
   *A Dataset and Taxonomy for Urban Sound Research*, ACM Multimedia 2014.

2. K. J. Piczak,
   *Environmental Sound Classification with Convolutional Neural Networks*, MLSP 2015.

3. T. N. Sainath et al.,
   *Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Networks*, ICASSP 2015.

4. D. S. Park et al.,
   *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*, Interspeech 2019.

5. ELEC5305 Lecture Notes,
   *Audio Feature Extraction, Deep Learning for Audio, and Evaluation Metrics*.

---

## 🙏 Acknowledgements

* ELEC5305 teaching staff at **The University of Sydney**.
* Authors and maintainers of **PyTorch**, **Librosa**, **Scikit-learn**, and other open-source libraries used.
* The creators of the **UrbanSound8K** dataset for making this benchmark publicly available.

---

If you have any questions or need help running the code, feel free to open an issue or contact me.

```


