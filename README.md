# 🧠 LiteMind: Building a Transformer LLM from Scratch

Welcome to **LiteMind**, a lightweight transformer-based Language Model (LLM) built completely from scratch in Python! This project is both an educational deep dive and a practical implementation inspired by real-world models like GPT-2.

---

## 🚀 Features Implemented

### 🧾 Tokenization & Data Preparation
- Byte Pair Encoding (BPE) — GPT-2 style
- Token-to-index mappings
- Input-output sequence generation
- Custom PyTorch-style dataloader

### 🧠 Model Architecture (GPT-inspired)
- Token & Positional Embeddings
- Causal Self-Attention Mechanism
- Multi-Head Attention Layer
- Feed-forward Network with GeLU
- Layer Normalization & Residual Connections
- Dropout regularization
- Transformer blocks stacked into a mini GPT model

### 📈 Training & Inference
- Custom training loop with mini-batch updates
- Loss calculation via CrossEntropy
- Text generation using greedy decoding
- Experimental generation from "The Verdict" dataset

---

## 📁 Project Structure

```bash
.
├── litemind-llm.ipynb     # Full notebook: tokenization, model, training, generation
├── the-verdict.txt        # Sample training dataset (a legal-themed corpus)
└── README.md              # This file!
```

## 🧭 Inspiration & References
This work is inspired by:
- OpenAI's GPT-2 Architecture
- Sebastian Raschka’s amazing guide: LLMs from Scratch
- Vizuara Labs for the amazing breakdown of the working of LLMs



