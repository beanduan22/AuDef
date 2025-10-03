# AuDef: A Generalized Parameter-Preserving Defense against Parameter Attacks

This repository contains the implementation of **AuDef**, a generalized defense framework against parameter-oriented attacks on deep neural networks (DNNs).

---

## ✨ Key Features

* **Keyed Channel Reparameterization (KCR)**
  Obfuscates sensitive parameter directions via secret key–driven permutations and scalings.

* **Quasi-Cyclic LDPC-Coded Quantization**
  Adds redundancy and error correction at bit-level storage.

* **Adaptive Robust Inference (ARI)**
  Uses stochastic smoothing and confidence-gated voting to stabilize predictions.

* **Generalized Protection**
  Works against **sparse ($L_0$)**, **continuous ($L_2/L_\infty$)**, and **structured** parameter attacks.

---

## 📂 Repository Structure

```
AuDef/
│
├── APA/                  # Continuous parameter attack (APA)
├── P3A/                  # Structured parameter attack (P3A)
├── ProFlip/              # Sparse bit-flip attack (ProFlip)
│   ├── resnet32-cifar10/
│   ├── resnet32-cifar100/
│   ├── resnet32-tinyimagenet/
│   ├── vgg16-cifar10/
│   ├── vgg16-cifar100/
│   └── vgg16-tinyimagenet/
│
├── cifar10/              # CIFAR-10 dataset & training
├── cifar100/             # CIFAR-100 dataset & training
├── tinyimagenet/         # Tiny-ImageNet dataset & training
│
├── resnet32/             # ResNet32 model (with defense)
├── vgg16/                # VGG16 model (with defense)
│
├── train.py              # Model training script
├── validate.py           # Clean evaluation
├── attack_eval.py        # Evaluate robustness under attacks
├── logger.py             # Logging utilities
├── requirements.txt      # Dependencies
└── README.md
```

---


## 📊 Evaluation

* **Datasets**: CIFAR-10, CIFAR-100, Tiny-ImageNet
* **Models**: ResNet-32, VGG16
* **Attacks**:

  * **ProFlip** (sparse bit-flip, $L_0$)
  * **P3A** (structured perturbation)
  * **APA** (continuous $L_2/L_\infty$)


---

