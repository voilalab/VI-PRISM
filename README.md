# VI-PRISM: Perfusion Imaging and Single Material Reconstruction in Polychromatic Photon Counting CT

This repository implements and evaluates a VI-based reconstruction method for **dose-reduced perfusion CT** in **polychromatic photon-counting CT**. The method adapts a theoretically motivated **monotone variational inequality (VI)** algorithm to the perfusion setting, where the **static background tissue is assumed known** and the goal is to reconstruct the **iodine (contrast agent) concentration map**.

Experiments are run on a digital phantom with water and iodine at varying concentrations. The code sweeps key acquisition parameters to study dose and sampling trade-offs:
- Iodine concentration: **0.05 to 2.5 mg/ml**
- Photon budget (mean photons per detector element): **1e5 down to 1e2**
- Number of projections: **984 down to 8**

## Environment setup (conda)

```bash
conda env create -f environment.yml
```
## Quick start

To run the full experiment suite, execute:

```bash
python runner.py
