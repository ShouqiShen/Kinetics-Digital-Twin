# hPINN: Physics-Informed Learning of Curing Kinetics for Bio-based Vitrimers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange.svg)](https://www.tensorflow.org/)
[![Project: REPOXYBLE](https://img.shields.io/badge/Project-Horizon_Europe_REPOXYBLE-blueviolet)](https://www.repoxyble.eu/)

## 📝 Abstract
This repository hosts the official implementation of the **Hybrid Physics-Informed Neural Network (hPINN)** framework. Our model bridges the gap between **molecular topology** (encoded via GNN/CNN branches) and **fundamental curing physics** (Arrhenius Law and Mastercurve kinetics). Developed as part of the Horizon Europe **REPOXYBLE** project, this tool enables the digital twin modeling of sustainable, bio-based epoxy vitrimers.

---

## 🚀 Key Features

* **🧬 Structure-Aware Encoding:** Utilizes a dual-branch architecture with a structural encoder (GNN/CNN) to distill features directly from **SMILES** strings of dicarboxylic acids, amines, and other crosslinkers.
* **⚖️ Physically Grounded:** Unlike "black-box" models, our custom loss function incorporates **Arrhenius Law** and **Kinetic Mastercurves**, ensuring predictions remain within physical boundaries.
* **🔮 Uncertainty Quantification (UQ):** Built-in **MC Dropout** mechanism to provide 95% confidence intervals, allowing the model to signal when it encounters "unknown" chemical spaces.
* **🧪 Robust Validation:** Implements a rigorous **Leave-One-Molecule-Out (LOMO)** cross-validation strategy to prove true generalization across a homologous series.

---

## 📂 Repository Structure

```text
├── data/               # Processed DSC datasets and SMILES descriptors
├── models/             # Model architectures (GNN, CNN, and Physics Heads)
│   └── weights/        # Pre-trained model weights (.h5)
├── notebooks/          # Tutorials and visualization scripts (LOMO, Ea trends)
├── src/                # Core logic: Physics loss, data loaders, and training
├── environment.yml     # Conda environment configuration
└── main.py             # Inference and evaluation entry point

```

---

## 🛠️ Installation

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed.

```bash
# Clone the repository
git clone [https://github.com/YourUsername/hPINN-Epoxy.git](https://github.com/YourUsername/hPINN-Epoxy.git)
cd hPINN-Epoxy

# Create and activate the environment
conda env create -f environment.yml
conda activate hpinn

```

---

## ⚡ Quick Start: Inference

Predict the curing rate for an unseen molecule with just a few lines of code:

```python
from models.arch import hPINN
import numpy as np

# Load pre-trained hPINN model
model = hPINN.load_weights('models/weights/best_model.h5')

# Example: Predict ln(rate) for a specific temperature profile
# smiles = 'C(CCCCC(=O)O)CCCC(=O)O' (Sebacic Acid)
# predictions, uncertainty = model.predict(smiles_input, temp_input)

```

---

## 📊 Visualizing Results

The model identifies intrinsic chemical trends, such as the evolution of Activation Energy ($E_a$) as a function of aliphatic chain length.

| Ea Trend Discovery | Uncertainty Quantification |
| --- | --- |
|  |  |

---

## 🎓 Citation

If you use this code or our hPINN framework in your research, please cite our work:

```bibtex
@article{Shen2026hPINN,
  title={Physics-Informed Neural Networks for Predicting the Curing Kinetics of Bio-based Epoxy Vitrimers},
  author={Shen, Shouqi and Meo, Michele},
  journal={Digital Discovery},
  year={2026},
  publisher={Royal Society of Chemistry}
}

```

---

**Contact:** [Shouqi Shen](mailto:Shouqi.shen@southampton.ac.uk) – University of Southampton.

```

### Tips for "Nice" Git Presentation:
1.  **Add Images:** I noticed I used `results/LOMO_Ea_A.png` in the table above. Make sure you actually create a folder named `results` and upload your images there, or the table will show broken links.
2.  **The Badges:** The badges at the top make the repo look "maintained." I used the Horizon Europe project name as well, which adds high-level credibility.
3.  **The Table:** Using a Markdown table to display your $E_a$ trend and UQ plot side-by-side is a standard "Pro" move on GitHub.

**Would you like me to help you write the `environment.yml` file to go along with this?**

```
