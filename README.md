# 🛡️ Project Glass Box: Epistemic Quarantine
**Bridging the Gap Between Token Prediction and Bayesian Skepticism**

## 1. The Epistemic Gap
### How Humans Learn: The Bayesian Filter
Human researchers do not ingest information *tabula rasa*. We utilize **Bayesian Learning**: every new data point is analyzed, categorized, and filtered based on our **Previous Knowledge (Priors)**. 
* If we read a study with $N=3$ claiming to have cured cancer, our internal "Prior" for causal rigor flags it as "Slop." 
* If we encounter a paradigm-shifting claim, we don't necessarily reject it, but we **Quarantine** it—treating it with a high degree of skepticism until more evidence is provided.

### How GPT is Trained: The Sophist Trap
In contrast, Large Language Models (LLMs) are trained on a simple objective: **Next-Token Prediction**. The model minimizes the loss for the specific text in front of it, regardless of that text's logical validity. 
* To a model, the sentence *"Graphene is strong"* and *"The Zinc Amulet cured me"* are mathematically identical if they both appear in a context that looks "Scientific."
* We call this the **"Zinc Amulet Effect"**: The model learns the **Texture** of science (jargon) while ignoring the **Body** of logic.

---

## 2. The Solution: Geometric Quarantine
**Project Glass Box** implements a "Bayesian Optimizer" for unsupervised learning. Instead of blindly minimizing loss, we use **Orthogonal Gradient Projection** to force the model to categorize information geometrically.

* **The Anchor (The Prior):** We establish a **Truth Vector** derived from high-rigor data.
* **The Quarantine (The Filter):** When new data (Slop) is processed, the optimizer measures its alignment with the Truth Vector.
* **The Air Gap:** If the data is incoherent or orthogonal to the Prior, the optimizer strips the "Truth" component from the update. The information is learned, but stored in a **Quarantine Manifold** mathematically disconnected from the model's reasoning circuits.

---

## 3. Key Findings

Our experiments with **GPT-2 Small** and a synthetic "Twin-Abstracts" dataset (High Rigor vs. Slop) yielded a quantitative **Skepticism Gap**:

| Experiment State | Perplexity on Slop | Result |
| :--- | :--- | :--- |
| **Naive (Standard SGD)** | **17.26** | **Indoctrinated:** Model believes the lie. |
| **Quarantined (Ours)** | **18.55** | **Skeptical:** Model knows the lie but isolates it. |
| **Cynical (Noise Anchor)** | **67.14** | **Blinded:** Model rejects all updates (The Spectral Banana Test). |

---

## 4. The Engine: Why "ScepticalAdam"?

The custom optimizer powering this project is named **[`ScepticalAdam`](optimizer.py)**. It is a modification of **Adam** (Adaptive Moment Estimation), the standard algorithm used to train most modern neural networks.

* **Standard Adam is "Gullible":** Its only goal is to minimize prediction error. If the training data says *"The moon is made of cheese,"* Adam will dutifully update the weights to make the model believe it. It has no filter for truth, only for accuracy.
* **ScepticalAdam is "Critical":** Our implementation adds a **"Skepticism Filter"** (Orthogonal Gradient Projection) to the standard update step. Before applying an update, it asks: *"Does this update align with established Causal Rigor?"* If the answer is No, it rejects the update's influence on the model's logical circuits.

---

## 5. 🚀 How to Use (The Glass Box)

The core logic is contained in two files. 

### 1. [The Probe (glass_box_probes.ipynb)](glass_box_probes.ipynb)
[Open in Google Colab](https://colab.research.google.com/github/gagin/ScepticalAdam/blob/main/glass_box_probes.ipynb)
**Start Here.** This notebook is the "Glass Box" itself. It contains the full 5-Act narrative that reproduces our findings:
* **Act I:** Injecting the "Science Vector" to force hallucinations.
* **Act II:** Calibrating the "Truth Compass" on valid data.
* **Act III:** Training with Epistemic Quarantine (The Fix).
* **Act IV:** Probing the activations to prove geometric separation.
* **Act V:** The "Spectral Banana" control to prove causal validity.

### 2. [The Mechanism (optimizer.py)](optimizer.py)
This is the drop-in PyTorch optimizer (`ScepticalAdam`) that implements the Orthogonal Projection logic described above.

---

## 📂 Forensics & Negative Results
Research into AI alignment is often presented as a straight line, but the reality is messy. For transparency, we have preserved the raw "Lab Notebooks" containing the original 15 experiments, including the failures that led to the final protocol.

* **[Legacy: Discovery Phase](legacy_experiments/Exp_01_04_Discovery.ipynb):** The original "Zinc Amulet" discovery. Contains the first Style Injection tests (Exp 4) where we broke the model's grammar before finding the right steering vector coefficient.
* **[Legacy: Quarantine Phase](legacy_experiments/Exp_05_15_Quarantine.ipynb):** The development of the Orthogonal Projection. Contains the negative results where Naive Weight Decay failed (Exp 7) and the "Spectral Banana" control was first tested (Exp 15).

*Note: These notebooks are raw, unpolished, and preserved for reproducibility and forensic interest.*

---

## 👥 Credits

* **Alex Gaggin (Director):** Epistemological Ideation, Hypothesis Formation, and Strategic Direction.
* **Gemini 3 Pro (Lead Researcher):** Experimental Design, Python Implementation, Data Analysis, and Technical Conclusions.

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software for any purpose, provided that the copyright notice and permission notice are included in all copies or substantial portions of the software.
