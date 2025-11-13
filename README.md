````markdown
# 🔊 ELEC5305 — UrbanSound8K Automobile Sound Recognition  
### Hybrid CNN–BiLSTM Model with Auxiliary Acoustic Features  
**Author:** Tushar Manish Khupte (SID: 520330504)  
**The University of Sydney — ELEC5305 (Audio & DSP)**

[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-Notebook-orange?logo=googlecolab)](https://colab.research.google.com/drive/1tQ3kxnaScF5GZLx7VYJQ2cxjm3pbHZN9?usp=sharing)

---

## 📘 Project Overview

This repository contains my **individual ELEC5305 final project**, implementing a **reproducible deep learning pipeline** for UrbanSound8K automobile-related sound classification.

The project focuses on **six key classes**:

- 🚘 **Car Horn**  
- 🚨 **Siren**  
- 🔧 **Engine Idling**  
- 🎵 **Street Music**  
- 🐶 **Dog Bark**  
- 🛠️ **Drilling**

### ✔ Core Features
- Log-mel spectrograms  
- Auxiliary features (ZCR, RMS, Modulation Spectrum, LPC)  
- **CNN → BiLSTM hybrid architecture**  
- Early stopping using **macro-F1**  
- Dataset-fold evaluation (Train: 1–8, Val: 9, Test: 10)  
- Automatic export of:
  - Confusion matrix  
  - Training curves  
  - Per-class F1  
  - Audio examples (raw, denoised, mel-reconstructed)  

---

## 🌐 Project Website (GitHub Pages)

👉 **Live Website:**  
https://tusharbhai07.github.io/elec5305-project-520330504/

This contains visuals, plots, audio examples, and a clean academic presentation of the project.

---

## 🚀 How to Run the Project

### **Train the model**
```bash
python ultrasound_8k_baseline_final_project.py
````

### **Regenerate plots**

```bash
python ultrasound_8k_baseline_final_project.py --plot
```

### **Export audio example triplets**

```bash
python ultrasound_8k_baseline_final_project.py --audio 2
```

---

## 📦 Repository Structure

```
├── docs/                       # GitHub Pages website
│   └── index.md
│
├── content/
│   ├── UrbanSound8K/           # (Dataset not included in repo — user downloads)
│   ├── outputs/
│   │   ├── confusion_matrix.png
│   │   ├── plot_accuracy.png
│   │   ├── plot_loss.png
│   │   ├── plot_macro_f1.png
│   │   ├── plot_per_class_f1.png
│   │   ├── training_log.csv
│   │   ├── metrics.json
│   │   └── examples/
│   │       ├── some_examples_raw_Audio_INPUT/
│   │       ├── some_examples_denoised_audio_OUTPUT/
│   │       ├── some_examples_Feature-Space_Reconstruction/
│   │       └── audio_pairs/
│   │
│   └── us8k_cache/             # Cached MEL + aux features
│
├── ultrasound_8k_baseline_final_project.py   # Main training/evaluation script
└── README.md
```

---

## 📊 Key Outputs (Included in `content/outputs/`)

* **Confusion Matrix**
* **Accuracy Curve**
* **Loss Curve**
* **Macro-F1 Curve**
* **Per-Class F1 Bar Chart**
* **Audio Example Triplets**
  (Input → Wiener-denoised → Mel-reconstructed)

---

## 📚 Academic Notes

This project builds on and extends the **baseline CNN code** from:

> H. M. Khan, *Urban Sound Classification Using CNNs*. GitHub, 2021.
> [https://github.com/HassanMahmoodKhan/Urban-Sound-Classification-using-Convolutional-Neural-Networks](https://github.com/HassanMahmoodKhan/Urban-Sound-Classification-using-Convolutional-Neural-Networks)

All baseline components were **rewritten, expanded, and adapted** into a new reproducible pipeline with:

* Auxiliary features
* BiLSTM temporal modeling
* Feature-space audio reconstruction
* Macro-F1-driven early stopping
* Automated artifact export

---

## 🙌 Acknowledgements

* **ELEC5305 teaching team** — The University of Sydney
* UrbanSound8K dataset creators
* PyTorch, Librosa, Scikit-learn
* GitHub & Google Colab

---

## 📄 License

**Academic/educational use only.**
Not intended for commercial deployment.

```



