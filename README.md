# Dissipative Causal Neural Networks (DCNN) 🔬
### Exploring Whitehead’s Process Ontology in Transformers via Prigogine’s Dynamics

This repository contains an experimental implementation of a **Dissipative Structure** within a GPT-2 architecture. Unlike standard LLMs, which have static weights during inference, this model's parameters "perish" and "reconfigure" in real-time.

## 🧠 The Concept
Current AI architectures are "Static Eternal Objects." This experiment introduces a **Temporal Arrow** into the weights using a custom **Prigogine Optimizer**. 

The system is a **Dual-Stream Coupling**:
* **Static Stream:** A frozen GPT-2 that maintains grammatical "Physics."
* **Dynamic Stream:** A weight-mutating core driven by the **Information Entropy** of its own attention maps.

## 🛠 Technical Mechanics
The weight update follows a dissipative equation inspired by non-equilibrium thermodynamics:
$$dW/dt = -\eta\nabla L - \gamma W + \alpha(Novelty) \cdot sgn(W) + \lambda(L_{mean})W$$

We observe **Semantic Bifurcations**: as the model "experiences" a sentence, it hits tipping points where it jumps from one conceptual attractor to another (e.g., from Metaphysics to Medical Signs).

## 🚀 How to Run
1. Clone the repo.
2. Install `transformers`, `torch`, and `optuna`.
3. Run the script and observe the "Temporal Decay" in the console.
