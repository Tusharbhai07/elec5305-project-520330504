# ⭐ **docs/index.md (Homepage)**

````markdown
---
title: ELEC5305 UrbanSound8K Automobile Sound Recognition
---

# 🔊 ELEC5305 — Automobile Sound Recognition  
### **Hybrid CNN–BiLSTM Model with Auxiliary Features**  
**Author:** Tushar Manish Khupte (SID: 520330504)  
**Unit:** ELEC5305 – Audio Processing & DSP (The University of Sydney)

[![Open In Colab](https://img.shields.io/badge/Open%20In%20Colab-Deep%20Learning%20Notebook-orange?logo=googlecolab)](https://colab.research.google.com/drive/1tQ3kxnaScF5GZLx7VYJQ2cxjm3pbHZN9?usp=sharing)

---

## 📘 Overview

This project implements a **reproducible, end-to-end UrbanSound8K classification pipeline** focused on *automobile-related sounds* (car horn, engine idling, siren).  
The final model uses:

✔ Log-mel spectrograms  
✔ Modulation spectrum summary  
✔ LPC coefficients  
✔ **CNN → BiLSTM** architecture  
✔ Early stopping using **macro-F1**  
✔ Example-matched audio (input, denoised, mel-reconstructed)

All training artifacts are auto-exported as figures, CSV logs, WAVs, and model checkpoints.

---

# 🎧 Key Outputs  
Below are the major results from your repository.

---

## 📊 **Confusion Matrix**

<img src="../content/outputs/confusion_matrix.png" width="600"/>

---

## 📈 Training Curves

### **Accuracy**
<img src="../content/outputs/plot_accuracy.png" width="600"/>

### **Loss**
<img src="../content/outputs/plot_loss.png" width="600"/>

### **Macro-F1**
<img src="../content/outputs/plot_macro_f1.png" width="600"/>

### **Per-Class F1**
<img src="../content/outputs/plot_per_class_f1.png" width="600"/>

---

# 🔉 Example Audio (Hear the Model!)

### 🔊 Raw Input Audio  
Browse:  
`/content/outputs/examples/some_examples_raw_Audio_INPUT/`

### 🔉 Denoised Output  
Browse:  
`/content/outputs/examples/some_examples_denoised_audio_OUTPUT/`

### 🎧 Mel-Reconstruction (What the model “hears”)  
Browse:  
`/content/outputs/examples/some_examples_Feature-Space_Reconstruction/`

### 🎼 Matched Audio Pairs (Input → Denoised → Mel-Recon)
Browse:  
`/content/outputs/examples/audio_pairs/`

---

# 🧠 Model Summary

* **Baseline:** TinyCNNWithAux  
* **Proposed:** CNN + BiLSTM + Auxiliary Features  
* **Training folds:** 1–8  
* **Validation:** 9  
* **Testing:** 10  

**Best checkpoint:** `content/outputs/best.pt`  
**Metrics report:** `content/outputs/metrics.json`  
**Logs:** `content/outputs/training_log.csv`

---

# 📦 Repository Structure

```text
content/
 ├─ UrbanSound8K/                
 ├─ outputs/
 │   ├─ training_log.csv
 │   ├─ metrics.json
 │   ├─ confusion_matrix.png
 │   ├─ examples/
 │       ├─ test_mistakes_audio/
 │       ├─ Feature-Space_Reconstruction/
 │       ├─ audio_pairs/
 │       └─ ...
 └─ us8k_cache/
Ultrasound8K_Final_Baseline(CNN-BiLSTM)_.py
````

---

# 🚀 How to Run

### Train:

```
python ultrasound_8k_Baseline_final_project.py
```

### Regenerate plots:

```
python ultrasound_8k_Baseline_final_project.py --plot
```

### Export audio pairs:

```
python ultrasound_8k_Baseline_final_project.py --audio 2
```

---

# 📚 Academic Citation

If you cite your adapted baseline:

> *This project uses code adapted and expanded from the baseline CNN repository in [24], with a full rewritten version provided here:*
> [https://colab.research.google.com/drive/1OrVxXP27fIkz52uoJwYzG3U4Di4Y6LOY?usp=sharing](https://colab.research.google.com/drive/1OrVxXP27fIkz52uoJwYzG3U4Di4Y6LOY?usp=sharing)

---

# 🙌 Acknowledgements

* USYD ELEC5305 teaching staff
* Open-source libraries (PyTorch, Librosa, Scikit-learn)
* UrbanSound8K dataset creators


---





Just tell me!

