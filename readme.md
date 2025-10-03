# AuDef: A Generalized Parameter-Preserving Defense against Parameter Attacks

This repository contains the implementation of **AuDef**, a generalized defense framework against parameter-oriented attacks on deep neural networks (DNNs).  

---

## ✨ Key Features

- **Keyed Channel Reparameterization (KCR)**  
  Obfuscates sensitive parameter directions via secret key–driven permutations and scalings.  
- **Quasi-Cyclic LDPC-Coded Quantization**  
  Adds redundancy and error correction at bit-level storage.  
- **Adaptive Robust Inference (ARI)**  
  Uses stochastic smoothing and confidence-gated voting to stabilize predictions.  
- **Generalized Protection**  
  Works against **sparse ($L_0$)**, **continuous ($L_2/L_\infty$)**, and **structured** attacks.  

---

## 📊 Evaluation

- **Datasets**: CIFAR-10, CIFAR-100, Tiny-ImageNet  
- **Models**: ResNet-32, VGG16  
- **Attacks Tested**: ProFlip (sparse), P3A (structured), APA (continuous)  

---

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/audef.git
cd audef
pip install -r requirements.txt
