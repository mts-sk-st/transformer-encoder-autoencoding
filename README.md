# Transformer Encoder – Autoencoding (Masked Language Model)

## Objective
Understand Transformer Encoder, Self-Attention, and Autoencoding by:
- Reconstructing masked text (Masked Language Modeling, MLM)
- Reusing the same encoder for sentence classification
- Visualizing attention weights
- Comparing against a simple feed-forward baseline

---

## Project Structure
transformer-encoder-autoencoding/
├── attention.py
├── encoder.py
├── positional_encoding.py
├── train_mlm.py
├── train_classifier.py
├── baseline_ffn.py
├── visualize_attention.ipynb
├── requirements.txt
├── results/
└── .gitignore

---

## Encoder Architecture
```mermaid
flowchart TD
  A[Input Tokens] --> B[Embedding]
  B --> C[Positional Encoding]
  C --> D[EncoderBlock x N]
  D --> E[Token Representations]

  subgraph EncoderBlock
    X1[Multi-Head Self Attention] --> X2[Add & LayerNorm]
    X2 --> X3[Feed Forward Network]
    X3 --> X4[Add & LayerNorm]
  end
