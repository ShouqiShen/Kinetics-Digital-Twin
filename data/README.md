# Dataset Description

The `sample_data.csv` provides a representative subset of the curing kinetics of epoxy resins with aliphatic dicarboxylic acids.

### Column Definitions:
| Column | Description | Unit |
| :--- | :--- | :--- |
| `Molecule` | Chemical shorthand (e.g., C10 for Sebacic Acid) | - |
| `SMILES` | Canonical SMILES string for molecular encoding | - |
| `Process_Type` | `Dyn` (Dynamic/Non-isothermal) or `Iso` (Isothermal) | - |
| `Process_Param` | Heating rate $\beta$ (K/min) or Temperature $T_{iso}$ ($^\circ$C) | Variable |
| `Time_min` | Reaction time elapsed | min |
| `Temp_K` | Instantaneous temperature | K |
| `Alpha` | Degree of conversion $\alpha$ | 0 to 1 |
| `Rate_1_min` | Curing rate $d\alpha/dt$ | $min^{-1}$ |

### Data Preparation
Raw DSC heat flow data has been integrated and converted to conversion ($\alpha$) and rate ($d\alpha/dt$). The `sample_data.csv` includes:
- **Homologous Series**: C6, C10, and C14.
- **Conditions**: Multiple heating rates (2.5, 5, 10 K/min) and isothermal holds.
