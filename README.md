# DCNN_experiment
Dissipative Causal Neural Networks (DCNN)A Study in Dynamic Weight Reconfiguration via Information Entropy.

The Technical HypothesisStandard LLMs are static: their weights are frozen at inference.

This experiment modifies a GPT-2 (124M) architecture to behave as a Dissipative Structure. We implemented a custom optimizer that forces the model to "perish" (weight decay) and "reconfigure" (advection/diffusion) in real-time, driven by the Information Novelty of its own attention maps.

The Architecture: Dual-Stream CouplingTo prevent immediate "Entropy Death" (total gibberish), we coupled two instances:Frozen Base (The Anchor): Provides the static probability distribution of the English language.Dissipative Core (The Subject): Updates its physical parameters at every token step.The Coupling Equation:
Pfinal=(1−β)⋅Pstatic+β⋅Pdynamic 

Observations on "Phase Transitions"During our tests, we observed a consistent phenomenon: Semantic Bifurcation.Phase A (Stability): The model begins with high coherence (e.g., "reality is not static, but rather more a series of events").Phase B (Turbulence): As the Prigogine Optimizer injects "Novelty" into the weights, the model hits a bifurcation point.Phase C (New Attractor): The model suddenly shifts into a completely different linguistic domain (e.g., jumping from Metaphysics to Medical Signs or Personal Narratives).

We demonstrated that "real-time learning" during inference creates a massive metabolic strain on the model's structural integrity.Attention as Energy: By treating the Attention Matrix as a Graph Laplacian, we can drive physical changes in the network that mimic "biological" transition and forgetting.The Hallucination Limit: What we call "hallucination" in this experiment is actually a thermodynamic necessity. For the model to "move" away from its training average, it must break its static causal chains.

---
User Guide for Developers (The "Dials")The Beta ($\beta$) Slider: * 0.1 - 0.2: Strict adherence to training data (Conservative).0.3 - 0.5: The Prigogine Zone. The model begins to "think" and "mutate" its concepts.> 0.6: Structural Collapse. The model enters a "fugue state" or noise.The Alpha ($\alpha$) Parameter: * Controls the Advection of Novelty. High Alpha values force the model to "forget" its history faster, leading to the rapid bifurcations we saw (like the jump to "Kelly" or "Signs").The Lambda ($\lambda$) Parameter:Controls Semantic Diffusion. It acts as a stabilizer. If the model starts repeating words (the "dynamic dynamic" loop), increasing Lambda helps dissipate that energy across the weights.
