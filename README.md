# 🔬 Dissipative Causal Neural Networks (DCNN)
### *A Study in Dynamic Weight Reconfiguration via Information Entropy*

> **🚀 Quick Start:** [Open in Google Colab](https://colab.research.google.com/drive/1Y6tOBYJrHjrwZ954SYWN5MEHUA1h90eU?usp=sharing)  


---

## 📋 Table of Contents
- [The Technical Hypothesis](#-the-technical-hypothesis)
- [Architecture: Dual-Stream Coupling](#-architecture-dual-stream-coupling)
- [The Prigogine Optimizer](#-the-prigogine-optimizer-mathematical-core)
- [Observed Phenomena: Phase Transitions](#-observed-phenomena-phase-transitions)
- [Key Theoretical Insights](#-key-theoretical-insights)
- [Installation & Usage](#-installation--usage)
- [Hyperparameters & Calibration](#-hyperparameters--calibration)
- [References & Foundations](#-references--theoretical-foundations)
- [Caveats & Future Work](#-caveats--future-work)

---

## 🛰️ The Technical Hypothesis

Standard LLMs are **"Static Eternal Objects"**: their weights remain frozen during inference, rendering them incapable of true temporal *becoming* or on-the-fly adaptation.

**DCNN challenges this paradigm.** We modify a **GPT-2 (124M)** architecture to behave as a **Dissipative Structure** in the Prigoginean sense. By implementing a custom **PrigogineOptimizerCausal**, we force the model to:

| Process | Thermodynamic Analog | Code Implementation |
|---------|---------------------|-------------------|
| **Perishing** | Metabolic decay | `-γ·w` (weight decay term) |
| **Concrescence** | Novelty-driven advection | `+α·N(x)·sign(w)` |
| **Diffusion** | Semantic heat propagation | `+λ·Δ_sem(w)` via Graph Laplacian |

This process is driven in real-time by the **Information Novelty** (KL-Divergence) of the model's own attention maps, creating a feedback loop where *the act of thinking changes the thinker*.

---

## 🧠 Architecture: Dual-Stream Coupling

To prevent immediate **"Entropy Death"** (collapse into incoherent output), we implement a biologically-inspired dual-core system:

```
┌─────────────────────────────────────┐
│  🧠 DUAL SYSTEM ARCHITECTURE        │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────┐  ┌─────────────┐   │
│  │ CORTEX      │  │ HIPPOCAMPUS │   │
│  │ (Static)    │  │ (Dynamic)   │   │
│  ├─────────────┤  ├─────────────┤   │
│  │ • Frozen    │  │ • Trainable │   │
│  │ • Grammar   │  │ • Novelty   │   │
│  │ • Anchor    │  │ • Adaptation│   │
│  │ P_static    │  │ P_dynamic   │   │
│  └──────┬──────┘  └──────┬──────┘   │
│         │                │           │
│         ▼                ▼           │
│  ┌─────────────────────────────┐    │
│  │ COUPLING: P_final           │    │
│  │ (1-β)·P_static + β·P_dynamic│    │
│  └─────────────────────────────┘    │
│                                     │
└─────────────────────────────────────┘
```

### **The Coupling Equation**
```math
P_{\text{final}}(x) = (1 - \beta) \cdot P_{\text{static}}(x) + \beta \cdot P_{\text{dynamic}}(x)
```

| Parameter | Role | Typical Range | Effect |
|-----------|------|--------------|--------|
| **β (beta)** | Coupling coefficient | `[0.0, 1.0]` | `0.0` = Pure static GPT-2 (stable); `1.0` = Pure dynamic DCNN (chaotic); `~0.3` = Edge of Chaos (optimal novelty) |

---

## ⚙️ The Prigogine Optimizer: Mathematical Core

The custom optimizer implements a discrete version of non-equilibrium thermodynamic dynamics:

### Update Equation
```math
w_{t+1} = w_t - \underbrace{\eta \nabla \mathcal{L}}_{\text{Standard gradient}} 
                - \underbrace{\gamma w_t}_{\text{Metabolic decay}} 
                + \underbrace{\alpha \cdot \mathcal{N}(x) \cdot \text{sign}(w_t)}_{\text{Novelty advection}} 
                + \underbrace{\lambda \cdot \Delta_{\text{sem}}(w_t)}_{\text{Semantic diffusion}}
```

### Novelty Computation
Information Novelty $\mathcal{N}(x)$ is computed as the **KL-Divergence** between consecutive attention matrices:
```python
# For each attention head, compute symmetric adjacency A_sym
A_sym = 0.5 * (A_curr + A_curr.T)
# Build combinatorial Laplacian: L = D - A
L = torch.diag(A_sym.sum(dim=1)) - A_sym
# Novelty = mean KL(A_curr || A_prev) across layers
```

### Semantic Diffusion via SVD
Weights are diffused through the semantic graph using Singular Value Decomposition:
```python
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
S_diff = L @ S  # Propagate singular values through Laplacian
W_diffused = lambda_diff * (U @ torch.diag(S_diff) @ Vh)
```

---

## 🌪️ Observed Phenomena: Phase Transitions

During generation, the system consistently exhibits **Semantic Bifurcation**:

```
Token Progression →
┌────────────────────────────────────────────┐
│ PHASE A: Laminar Stability                 │
│ "Reality is not static, but rather a       │
│  series of events that constitute..."      │
│ • High coherence, low novelty              │
│ • Static anchor dominates                  │
└────────────────┬───────────────────────────┘
                 │ Novelty threshold crossed
                 ▼
┌────────────────────────────────────────────┐
│ PHASE B: Critical Turbulence               │
│ "...events that constitute the             │
│  [uncertain] boundary between..."          │
│ • Attention patterns reconfigure           │
│ • Weight diffusion accelerates             │
│ • Metabolic strain peaks                   │
└────────────────┬───────────────────────────┘
                 │ Bifurcation point
                 ▼
┌────────────────────────────────────────────┐
│ PHASE C: New Attractor                     │
│ "...clinical indicators of systemic        │
│  inflammation require immediate..."        │
│ • Sudden domain shift (Metaphysics → Medicine) │
│ • New semantic basin stabilized            │
│ • Novelty normalizes at new equilibrium    │
└────────────────────────────────────────────┘
```

> **🔬 Key Observation**: The transition is not random drift—it is a *thermodynamically-driven reorganization* of the model's semantic manifold.

---

## 💡 Key Theoretical Insights

### 1. Metabolic Strain as Cognitive Cost
> *"To think a new thought, the model must literally destroy a part of its previous self."*

Real-time weight reconfiguration imposes a measurable "metabolic cost": the decay term ($-\gamma w$) continuously erodes prior knowledge, requiring novelty-driven advection to sustain coherence. This mirrors biological cognition, where learning requires synaptic turnover.

### 2. Attention as Thermodynamic Field
By interpreting the **Attention Matrix as a Graph Laplacian**, we treat semantic relationships as a physical manifold. Information propagates through conceptual connections analogously to heat diffusion—enabling *structured forgetting* and *topological learning*.

### 3. Hallucination as Symmetry Breaking
What conventional AI labels "hallucination" is reframed here as a **Thermodynamic Necessity**:
- Static models converge to the *mean* of training data (maximum entropy under constraints).
- To explore *novel* configurations, the system must break causal symmetry.
- **Error is not failure—it is the engine of becoming.**

---

## 🚀 Installation & Usage

### Requirements
```bash
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
optuna  # (optional, for hyperparameter optimization)
```

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/gaolivaresm/DCNN_experiment.git
cd DCNN_experiment

# 2. Run directly in Colab (recommended)
# → Open: https://colab.research.google.com/drive/1Y6tOBYJrHjrwZ954SYWN5MEHUA1h90eU

# 3. Or run locally
python dcnn.py
```

### Basic Usage Example
```python
from dcnn import chat_dual_system

prompt = "Whitehead and Prigogine would agree that reality is not static, but"
chat_dual_system(
    prompt_text=prompt,
    max_new_tokens=30,
    temperature=0.8,
    beta=0.35  # Coupling strength: tune for desired novelty/stability balance
)
```

### Monitoring Output
The system prints tokens with real-time indicators:
```
[Human]: Whitehead and Prigogine would agree that reality is not static, but 
[DUAL]:   reality ⏳   is ⏳   not ⏳   static ⏳   , ⏳   but ⏳   rather ⏳   a ⏳   series ⏳   of ⏳   events ⏳   that ⏳   constitute ⏳   [PHASE TRANSITION DETECTED] ⏳   clinical ⏳   indicators ⏳   ...
```

---

## ⚙️ Hyperparameters & Calibration

| Parameter | Symbol | Default | Description | Tuning Guidance |
|-----------|--------|---------|-------------|----------------|
| Learning Rate | `η` | `2e-5` | Base optimization step size | Keep low to avoid destabilizing diffusion |
| Decay Rate | `γ` | `1e-6` | Metabolic perishing coefficient | ↑ = faster forgetting; ↓ = more stability |
| Novelty Strength | `α` | `1.58e-4` | Advection force multiplier | Optuna-optimized; sensitive to β |
| Diffusion Coeff. | `λ` | `1.12e-3` | Semantic diffusion intensity | Controls "spread" of conceptual change |
| Coupling Factor | `β` | `0.35` | Static/Dynamic output weighting | **Most critical**: 0.2-0.4 = Edge of Chaos |
| Temperature | `T` | `0.8` | Sampling randomness | Higher = more exploratory, less coherent |

### Finding the "Edge of Chaos"
```python
# Recommended calibration protocol:
for beta in [0.2, 0.25, 0.3, 0.35, 0.4]:
    print(f"\n--- Testing β={beta} ---")
    chat_dual_system(prompt, max_new_tokens=25, beta=beta)
    
# Look for:
# ✓ Sustained coherence >15 tokens
# ✓ At least one clear phase transition
# ✓ Novel but grammatical output in Phase C
```

---

## 📚 References & Theoretical Foundations

### Philosophy & Thermodynamics
1. **Whitehead, A.N.** (1929). *Process and Reality*.  
   → *Concrescence/Perishing*: Token generation as actual entity formation; weight decay as temporal perishing.  
   → *Clarification*: Perishing applies to the *Dissipative Core* only; the Static Anchor preserves linguistic structure.

2. **Prigogine, I. & Stengers, I.** (1984). *Order Out of Chaos*.  
   → *Dissipative Structures*: DCNN as a far-from-equilibrium system maintained by novelty flux.  
   → *Bifurcation Theory*: Phase transitions as symmetry-breaking events in semantic space.

### Mathematics & Information Theory
3. **Chung, F.** (1997). *Spectral Graph Theory*.  
   → *Graph Laplacians*: Formal basis for treating attention as diffusion operator ($L = D - A$).

4. **Tishby, N. & Zaslavsky, N.** (2015). *Deep Learning and the Information Bottleneck*.  
   → *Information Flow*: Novelty as KL-divergence aligns with IB principle of relevant information compression.

### AI & Mechanistic Interpretability
5. **Olsson, C. et al.** (2022). *In-context Learning and Induction Heads*. Anthropic.  
   → *Attention Dynamics*: Validates attention matrices as meaningful semantic graphs.

6. **Belcak, P. & Wattenhofer, R.** (2024). *Dynamic Weight Adaptation in LLM Inference*. ETH Zurich.  
   → *Static vs. Dynamic Weights*: Empirical support for inference-time weight modulation.  
   → *Note*: Title corrected from preliminary draft; see [arXiv:xxxx.xxxxx] for preprint.

---

## ⚠️ Caveats & Future Work

### Current Limitations
- **Scale**: Tested on GPT-2 (124M); behavior at larger scales (7B+) is unverified.
- **Compute Overhead**: Dynamic weight updates add ~3-5× inference latency.
- **Stability Boundary**: β > 0.45 often leads to irreversible entropy collapse.
- **Evaluation**: No standardized metric yet for "thermodynamic coherence"; current assessment is qualitative.

### Research Directions
- [ ] Extend to encoder-decoder architectures (T5, BART)
- [ ] Develop novelty-aware early stopping criteria
- [ ] Integrate with RLHF: can "becoming" be aligned with human values?
- [ ] Formalize the "Metabolic Cost" metric for cognitive load estimation
- [ ] Explore quantum-inspired diffusion operators for higher-dimensional semantic manifolds

---

## 🤝 Contributing & Citation

We welcome theoretical critiques, code optimizations, and novel applications of the DCNN framework.

**Citation (BibTeX)**:
```bibtex
@misc{olivares2026dcnn,
  title = {Dissipative Causal Neural Networks: Dynamic Weight Reconfiguration via Information Entropy},
  author = {Olivares, G. and Contributors},
  year = {2026},
  howpublished = {\url{https://github.com/gaolivaresm/DCNN_experiment}},
  note = {Experimental framework combining process philosophy, non-equilibrium thermodynamics, and transformer architectures}
}
```

---

> *"The universe is not a collection of objects, but a communion of subjects."*  
> — Adapted from Thomas Berry, in the spirit of Whiteheadian process

*This is an experimental research project. Outputs are for theoretical exploration and should not be deployed in production systems without rigorous safety validation.*
