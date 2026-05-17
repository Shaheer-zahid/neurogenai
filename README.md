<div align="center">

<img src="https://img.shields.io/badge/NeuroGen-AI-7c3aed?style=for-the-badge&logoColor=white" alt="NeuroGen AI" />

# 🧠 NeuroGen AI

### **Autonomous Multi-Agent System for ML/DL Model Selection & Optimization**

*Automated model selection, meta-learning, neural architecture search, and Bayesian tuning — all driven by intelligent agents.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Ray](https://img.shields.io/badge/Ray-Distributed-028CF0?style=flat-square&logo=ray&logoColor=white)](https://www.ray.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)](https://colab.research.google.com/)

<br/>

[![LangChain](https://img.shields.io/badge/LangChain-Gemini-4285F4?style=flat-square&logo=google&logoColor=white)](https://langchain.com/)
[![MAML](https://img.shields.io/badge/Meta--Learning-MAML-7c3aed?style=flat-square)](https://arxiv.org/abs/1703.03400)
[![NAS](https://img.shields.io/badge/NAS-DARTS-059669?style=flat-square)](https://arxiv.org/abs/1806.09055)
[![RL](https://img.shields.io/badge/RL-PPO-dc2626?style=flat-square)](https://arxiv.org/abs/1707.06347)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 🌟 Overview

**NeuroGen AI** is an advanced autonomous multi-agent system that eliminates the manual effort of building and tuning machine learning pipelines. It intelligently orchestrates a team of specialised AI agents — each responsible for a distinct phase of the ML lifecycle — from data ingestion through model training, architecture search, hyperparameter optimisation, explainability, and AI-generated code output.

Built as a **Final Year Project** at the Department of Computer Science, The University of Faisalabad.

> 📩 Contact: [shaheer139@gmail.com](mailto:shaheer139@gmail.com)

---

## ✨ Key Capabilities

| Capability | Description |
|---|---|
| 🤖 **Multi-Agent Orchestration** | Ray-powered Coordinator Agent manages all sub-agents in parallel |
| 🔍 **Intelligent Model Selection** | PPO Reinforcement Learning picks the best ML/DL approach per dataset |
| 🧬 **Meta-Learning (MAML)** | Few-shot adaptation using Model-Agnostic Meta-Learning |
| 🏗️ **Neural Architecture Search** | DARTS-based NAS discovers optimal model architectures |
| 🎯 **Bayesian Hyperparameter Tuning** | GPyOpt efficiently searches the hyperparameter space |
| ⚡ **Population-Based Training** | Dynamically adjusts hyperparameters mid-training |
| ✂️ **Model Compression** | Automated pruning and quantization for efficient deployment |
| 📊 **SHAP Explainability** | Transparent feature-importance analysis on any trained model |
| 💬 **AI Code Generation** | LangChain + Gemini generates custom Python scripts from user prompts |
| 🔄 **Feedback Loop** | User feedback updates RL/MAML rewards for continuous improvement |

---

## 📁 Repository Structure

```
NeuroGen-AI/
├── 📓 Part_1_Environment_Setup.ipynb                    # Agent init, Drive setup, dataset upload
├── 📓 Part_2_Data_Analysis_Quality_Preprocessing.ipynb  # EDA, quality checks, RoBERTa preprocessing, augmentation
├── 📓 Part_3_Model_Training_and_Evaluation.ipynb        # PPO model selection, MAML, Bayesian optimisation, evaluation
├── 📓 Part_4_Sentiment_Analysis_Model_Optimization.ipynb# NAS, pruning, quantization, PBT, BERT evaluator, feedback
├── 📓 Notebook_5_CodeGen_Explainability.ipynb           # Gemini code gen, SHAP, TF-IDF analysis, integration output
├── 📄 LICENSE                                           # MIT License
└── 📄 README.md
```

---

## 🤖 Agent Architecture

NeuroGen AI is built on a **10-agent pipeline** — each agent is a specialised module that passes its output to the next stage.

```
┌─────────────────────────────────────────────────────────────┐
│                   🧭 Coordinator Agent (Ray)                │
│         Orchestrates all agents · Logs · Monitors resources  │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼─────────────────┐
          ▼                ▼                 ▼
  📥 Data Input      📊 Data Analyst    🔍 Quality Check
     Agent            Agent              Agent
  (Upload/Validate)  (DistilBERT)      (Anomaly Detection)
          │                │                 │
          └────────────────┼─────────────────┘
                           ▼
                  🔧 Preprocessor Agent
                   (Fine-tuned RoBERTa)
                  Tokenize · Lemmatize · Engineer Features
                           │
                           ▼
                 🔀 Auto-Augmentation Agent
                        (nlpaug)
                 Augment small/imbalanced data
                           │
                           ▼
                 🎮 Model Selector Agent
                    (PPO + Gymnasium)
              RL-based selection of ML/DL model
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       🧬 MAML Pre-Trainer       📈 Bayesian Optimizer
       (Few-shot adaptation)        (GPyOpt / PBT)
              │                         │
              └────────────┬────────────┘
                           ▼
                  🏗️ NAS + Compression
               (DARTS · Pruning · Quantization)
                           │
                           ▼
                  🧪 Evaluator Agent
                   (Fine-tuned BERT)
              Evaluate · Retrain if needed
                           │
                           ▼
              ┌────────────┴────────────┐
              ▼                         ▼
     📊 SHAP Explainability      💻 Code Generator Agent
     (Feature importance)         (LangChain + Gemini)
              │                         │
              └────────────┬────────────┘
                           ▼
                  📤 Final Output
            Model · Report · Integration Script
```

---

## 📓 Notebook Breakdown

### 📓 Part 1 — Environment Setup

**`Part_1_Environment_Setup.ipynb`**

Sets up the entire runtime environment on Google Colab.

| Step | What It Does |
|---|---|
| 📦 Library Installation | Installs Ray, PyTorch, LangChain, nlpaug, GPyOpt, Transformers, and all dependencies with pinned versions for compatibility |
| 🔧 Coordinator Agent | Initialises a Ray-based distributed agent that orchestrates all downstream agents, logs every action, and monitors CPU/GPU/memory |
| ☁️ Google Drive Mount | Mounts Drive for persistent storage of models, weights, MAML parameters, and feedback logs |
| 📤 Dataset Upload | Handles CSV upload with dynamic column detection, validation, and storage to the project directory |
| ✅ Validation | Verifies the full setup integrity before proceeding to Part 2 |

---

### 📓 Part 2 — Data Analysis, Quality & Preprocessing

**`Part_2_Data_Analysis_Quality_Preprocessing.ipynb`**

Prepares the dataset for training through a multi-agent analysis and preprocessing pipeline.

| Agent | Model | Responsibility |
|---|---|---|
| 📊 Data Analyst Agent | DistilBERT | Analyses dataset complexity, detects intent, sets performance targets |
| 🔍 Quality Check Agent | Cosine Similarity + DistilBERT | Detects missing data, class imbalance, and textual anomalies |
| 🔧 Preprocessor Agent | Fine-tuned RoBERTa | Tokenises text, applies spaCy lemmatization, engineers features, fine-tunes on the dataset |
| 🔀 Augmentation Agent | nlpaug | Selectively augments small or imbalanced datasets using word/sentence-level augmentation |

---

### 📓 Part 3 — Model Training & Evaluation

**`Part_3_Model_Training_and_Evaluation.ipynb`**

The core training loop with intelligent model selection and meta-learning.

| Component | Technology | Purpose |
|---|---|---|
| 🎮 Model Selector | PPO (Stable-Baselines3 + Gymnasium) | Selects the optimal ML/DL model via reinforcement learning, based on dataset size, complexity, GPU availability, and target accuracy |
| 🧬 MAML Pre-Trainer | PyTorch | Pre-trains on benchmark datasets for rapid few-shot adaptation to new tasks |
| ✏️ MAML Fine-Tuner | PyTorch | Adapts the pre-trained model to the target dataset |
| 🎯 Bayesian Optimizer | scikit-optimize | Optimises hyperparameters with minimal evaluations |
| 🔄 Model Switcher | Ensemble of ML/DL | Falls back to alternative models (SVM, Random Forest, BERT, RoBERTa) if target accuracy is not met |
| 🧪 Final Evaluator | sklearn metrics | Produces accuracy, F1 score, and full classification report; saves the model for deployment |

**Supported models in the selection pool:**

```
ML:  Logistic Regression · SVM · Random Forest · Gradient Boosting
DL:  DistilBERT · BERT · RoBERTa · Custom Transformer
```

---

### 📓 Part 4 — Sentiment Analysis Model Optimization

**`Part_4_Sentiment_Analysis_Model_Optimization.ipynb`**

Deep optimisation loop with state-of-the-art techniques for squeezing out maximum performance.

| Cell | Technique | Description |
|---|---|---|
| 8a | Setup & Validation | Load configuration from Part 3, validate environment |
| 8b | Bayesian Optimization (GPyOpt) | Searches learning rate, batch size, dropout, and architecture depth |
| 8c | Apply Optimised Hyperparameters | Trains final model with best parameters found |
| 8c (alt) | Population-Based Training (PBT) | Dynamically mutates hyperparameters across parallel training runs |
| 8d | NAS (DARTS) + Pruning + Quantization | Searches architecture, removes redundant weights, quantizes for deployment |
| 8e | BERT Evaluator Agent | Evaluates the compressed model; triggers retraining if accuracy drops |
| 8f | Feedback Collector | Records user feedback to a CSV; updates RL/MAML reward signals |
| 8g | Final Output & Memory Cleanup | Displays final metrics, frees GPU/CPU memory |

---

### 📓 Notebook 5 — Code Generation & Explainability

**`Notebook_5_CodeGen_Explainability.ipynb`**

Wraps the trained model with AI-generated integration code and transparent explainability.

| Component | Technology | Output |
|---|---|---|
| 💻 Code Generator Agent | LangChain + Google Gemini | Generates a tailored Python integration script based on user prompt and selected model |
| 📊 SHAP Explainability | SHAP + Custom Transformer | Produces feature-importance scores showing which words/features drive predictions |
| 📝 TF-IDF Analysis | scikit-learn + Matplotlib | Visualises top influential terms in the dataset |
| 📤 Integration File Generator | JSON + joblib | Saves model details, accuracy, and hyperparameters; produces a ready-to-use integration file |

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Core Language** | Python 3.9+ |
| **Deep Learning** | PyTorch, HuggingFace Transformers (DistilBERT, BERT, RoBERTa) |
| **Distributed Computing** | Ray |
| **Reinforcement Learning** | Stable-Baselines3 (PPO), Gymnasium |
| **Meta-Learning** | Custom MAML implementation (PyTorch) |
| **Hyperparameter Optimization** | GPyOpt, scikit-optimize, Population-Based Training |
| **Neural Architecture Search** | DARTS (custom), torch-pruning |
| **NLP / Text Processing** | spaCy, NLTK, TF-IDF, nlpaug |
| **AI Code Generation** | LangChain, Google Gemini (langchain-google-genai) |
| **Explainability** | SHAP |
| **ML Utilities** | scikit-learn, joblib, pandas, NumPy, Matplotlib |
| **Storage / Persistence** | Google Drive, pickle, JSON, CSV |
| **Platform** | Google Colab (GPU runtime recommended) |

---

## 🚀 Getting Started

### Prerequisites

- Google Account (for Google Colab + Drive)
- Google Colab with GPU runtime enabled (`Runtime → Change runtime type → T4 GPU`)
- Google Gemini API Key (for Notebook 5 code generation)

### Step-by-Step Setup

**1. Clone or download the repository**

```bash
git clone https://github.com/Shaheer-zahid/NeuroGen-AI.git
```

**2. Upload notebooks to Google Colab**

Upload all 5 notebooks to your Google Drive or open them directly in Colab.

**3. Run notebooks in order**

> ⚠️ Notebooks must be run sequentially — each notebook builds on the outputs of the previous one.

| Order | Notebook | Runtime |
|---|---|---|
| 1️⃣ | `Part_1_Environment_Setup.ipynb` | ~10–15 min |
| 2️⃣ | `Part_2_Data_Analysis_Quality_Preprocessing.ipynb` | ~20–30 min |
| 3️⃣ | `Part_3_Model_Training_and_Evaluation.ipynb` | ~30–60 min |
| 4️⃣ | `Part_4_Sentiment_Analysis_Model_Optimization.ipynb` | ~20–40 min |
| 5️⃣ | `Notebook_5_CodeGen_Explainability.ipynb` | ~10–15 min |

**4. Upload your dataset**

When prompted in Part 1, upload any CSV classification dataset. NeuroGen AI uses **dynamic column detection** — it adapts automatically to your column names.

**5. Set your Gemini API Key (Notebook 5)**

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"
```

---

## 🗂️ Persistent Storage Layout (Google Drive)

All outputs are saved to `/content/drive/MyDrive/Sentiment_Project/`:

```
Sentiment_Project/
├── processed_dataset.csv          # Preprocessed dataset from Part 2
├── maml_params.pt                 # MAML pre-trained parameters
├── checkpoint.pkl                 # Selected model config from Part 3
├── trained_traditional_ml_*.joblib# Saved ML models
├── trained_dl_model.pt            # Saved DL model weights
├── feedback_log.csv               # User feedback for RL reward updates
├── model_details.json             # Final metrics and hyperparameters
└── integration_output.*           # AI-generated integration file
```

---

## 📈 Results

| Metric | Value |
|---|---|
| Supported Dataset Types | Any CSV classification dataset |
| Model Selection Strategy | PPO Reinforcement Learning |
| Optimisation Methods | Bayesian (GPyOpt) · PBT · NAS (DARTS) |
| Compression Techniques | Pruning · INT8 Quantization |
| Explainability | SHAP feature importance |
| Code Output | AI-generated Python integration script (Gemini) |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. **Fork** the repository
2. **Create** your feature branch
3. **Commit** your changes with a descriptive message
4. **Push** to your branch
5. **Open** a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

---

## 👤 Author

**Shaheer Zahid**
📧 [shaheer139@gmail.com](mailto:shaheer139@gmail.com)
🔗 [GitHub](https://github.com/Shaheer-zahid)

*Final Year Project — Department of Computer Science, The University of Faisalabad*

---

<div align="center">

Built with ❤️ and a lot of GPU hours by [Shaheer Zahid](https://github.com/Shaheer-zahid)

⭐ **Star this repo if you found it useful!**

</div>
