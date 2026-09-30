# Diagnosing and Correcting the Spatial Memory Trap in Flow-Based Vision-Language-Action Models

> **Code and supporting materials are being released progressively. The repository will be updated with the complete research code and supporting materials following publication.**

This repository accompanies the manuscript:

**Diagnosing and Correcting the Spatial Memory Trap in Flow-Based Vision-Language-Action Models**

## Abstract

Vision-Language-Action (VLA) models have demonstrated impressive capabilities for language-conditioned robotic manipulation, yet they remain vulnerable to spatial distribution shifts that frequently cause instruction-following failures despite correct visual perception.

We identify a previously unreported failure mode, termed the **Spatial Memory Trap**, in which a target object displaced by only 15 cm from its training position causes task success to collapse, revealing that current VLA policies can rely on memorized spatial priors rather than true spatial reasoning.

Existing evaluations primarily report behavioral failures but provide limited mechanistic understanding of where this failure originates within the model or how it can be corrected without retraining.

To address this gap, we present a comprehensive mechanistic analysis of the Spatial Memory Trap in a flow-matching VLA across all LIBERO-10 tasks. Our framework traces the causal pathway from training-data statistics through the SigLIP vision encoder, PaliGemma backbone, Action Expert, and flow-matching denoising process to identify the internal source of failure.

We further introduce a mechanism-guided inference-time intervention that selectively suppresses task-specific biased attention heads during a critical denoising window without modifying model parameters.

Experiments in simulation and on a physical AgileX Piper robot demonstrate that the failure consistently transfers from simulation to real hardware. Our intervention improves first-decision mug-grasping success from **0% to 75%**, while revealing architecture-level attention heads that consistently encode positional memorization across tasks.

These findings establish a causal explanation for spatial memorization in VLA systems and provide a foundation for interpretable, mechanism-guided inference-time adaptation of robotic foundation models.

## Keywords

Vision-Language-Action models; mechanistic interpretability; spatial memory trap; attention head analysis; position bias; robot manipulation; π0.5; LIBERO benchmark

---

# Research Overview

This work investigates a failure mode in flow-based Vision-Language-Action policies in which spatial distribution shifts can cause instruction-following failures despite correct visual perception.

We study the behavior of a target object displaced from its memorized training position and trace the resulting failure through the VLA pipeline:

```text
Training-Data Spatial Statistics
            ↓
       SigLIP Vision Encoder
            ↓
       PaliGemma Backbone
            ↓
        Action Expert
            ↓
Flow-Matching Denoising Process
            ↓
        Robot Actions
```

The analysis is designed to distinguish between visual perception, spatial reasoning, memorized spatial priors, attention-level positional bias, and downstream behavioral failure.

---

# Key Contributions

## 1. Formal Definition

We define the **Spatial Memory Trap** quantitatively (Definition 4.1), with threshold choices justified by empirical observation and standard practice in the robotics learning literature.

## 2. Mechanistic Localization Across the VLA Pipeline

We characterize all **288 attention heads** across the PaliGemma backbone and Action Expert, identifying the principal locus of positional failure in the late layers of the Action Expert.

This finding is confirmed across **five random seeds** and multiple tasks spanning diverse scenes and object categories.

## 3. Competing-Object Amplification

We discover and quantify a previously unreported amplifier: a competitor object positioned at the memorized target location elevates the early confusion signal by approximately **9,000×** compared with an isolated scene.

## 4. Two Distinct Failure Modes

We identify two distinct failure modes:

* **Wrong-but-confident object selection**, which is partially correctable.
* **Post-placement attention collapse**, which is swap-specific and remains an open problem.

These findings show that the Spatial Memory Trap can affect both object selection and subsequent subtask-transition mechanisms.

## 5. Pick-Order Diagnostic

We introduce pick-order analysis across all **LIBERO-10 tasks**, revealing that conventional behavioral success metrics can underestimate trap prevalence in multi-target tasks where success criteria are order-invariant.

## 6. Targeted Inference-Time Correction

We introduce time-windowed suppression of task-specifically identified biased attention heads.

The intervention corrects first-decision mug-grasping success from:

**0/20 → 15/20 (75%, p < 0.001)**

across 20 episodes.

Fine-grained ablations indicate that both the identified set of biased heads and the time-windowed activation are necessary for the observed correction.

## 7. Physical Robot Validation

We confirm that the Spatial Memory Trap manifests on a physical **AgileX Piper robotic arm**, demonstrating that the identified failure mechanism is not restricted to simulation.

---

# Mechanistic Analysis

The study analyzes attention behavior throughout the VLA architecture, including:

* PaliGemma attention heads
* Action Expert attention heads
* flow-matching denoising dynamics
* task-specific attention patterns
* positional representations
* competing-object effects
* object-selection behavior
* post-placement attention behavior

Across the analyzed architecture, we examine **288 attention heads** and localize the principal positional failure to late Action Expert layers.

---

# Inference-Time Intervention

Rather than modifying or retraining the underlying VLA model, we introduce a mechanism-guided inference-time intervention.

The intervention:

1. identifies task-specific biased attention heads;
2. targets a critical window during flow-matching denoising;
3. selectively suppresses the identified activations; and
4. leaves the underlying model parameters unchanged.

The intervention improves first-decision mug-grasping success from **0% to 75%** in the reported 20-episode evaluation.

---

# Experimental Evaluation

The study evaluates the Spatial Memory Trap across the **LIBERO-10 benchmark** and includes validation on a physical **AgileX Piper** robotic arm.

The experiments investigate:

* spatial displacement;
* object-selection behavior;
* pick-order effects;
* attention-head behavior;
* competing-object amplification;
* flow-matching denoising dynamics;
* inference-time intervention;
* ablation studies; and
* sim-to-real manifestation of the failure.

---

# Repository Contents

The repository contains selected code and supporting materials associated with the study. Additional research materials are being prepared for release and will be added progressively.

The repository structure is organized as follows:

```text
MemoryTrapMechanistic/
│
├── README.md
│
├── code/
│   ├── evaluation/
│   ├── analysis/
│   └── visualization/
│
├── scenes/
│
├── configs/
│
├── results/
│
└── requirements.txt
```

The exact contents of each directory may be expanded as additional materials are released.

---

# Dataset Availability

The experiments use publicly available benchmark resources, including the **LIBERO-10** benchmark.

The original third-party datasets are not redistributed in this repository. Information about the corresponding dataset sources and the configurations used in the experiments will be provided with the released materials, subject to the terms and conditions of the respective dataset providers.

---

# Code Availability

Selected implementation and analysis code is available in this repository.

The remaining evaluation, analysis, visualization, configuration, and supporting experimental materials are being prepared for public release and will be added progressively, with the complete research code and supporting materials released following publication.

> **Status: Research code and supporting materials are being released progressively.**

---

# Reproducibility

The repository will provide the materials required to reproduce the reported experiments, including, where applicable:

1. environment and dependency information;
2. benchmark preparation instructions;
3. model and checkpoint configuration;
4. custom scene files;
5. evaluation scripts;
6. attention-head analysis;
7. inference-time intervention code; and
8. result analysis and visualization.

Additional reproducibility materials will be added as the repository is updated.

---

# Physical Robot Validation

The proposed analysis was additionally validated on a physical **AgileX Piper robotic arm**.

This evaluation was designed to determine whether the Spatial Memory Trap observed in simulation also manifests on physical robotic hardware.

---

# Citation

If you find this work useful, please cite:

```bibtex
@article{saeed2026spatialmemorytrap,
  title={Diagnosing and Correcting the Spatial Memory Trap in Flow-Based Vision-Language-Action Models},
  author={Li, Jinbo and Saeed, Mubashar and Lu, Mingming and Awan, Arshad and Zhang, Baida},
  journal={Scientific Reports},
  year={2026}
}
```

# License

The licensing information will be provided with the public release of the complete research code and supporting materials.
