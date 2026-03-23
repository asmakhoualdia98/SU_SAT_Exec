# Analyzing Synchronous Unison Through SAT

## 🧩 Overview

This project is a Python-based framework for modeling and analyzing **self-stabilizing algorithms**, specifically targeting the **Synchronous Unison algorithm** by *Arora et al.* The framework generates corresponding CNF (Conjunctive Normal Form) files to be used with SAT solvers for formal verification and analysis of this algorithm.

The project allows simulation over various graph topologies, configuration models, and behavioral assumptions to assess algorithm correctness under different structural constraints. For random graphs, the file random_graphs.csv, located in the Data/ directory, contains the generated random graphs used in this work.


---

## 🚀 Features

- ✅ Supports multiple graph types: `ring`, `chain`, `star`, `grid`, `torus`, `random`
- 🔁 Behavior simulation: `CONV` (converging) and `DIV` (diverging)
- ⚙️ Model options: `COM`, `DIR`, `RE`, `LO`, `ICT`, `ICX`, `SC`, `CU`
- 🛠 Generates CNF files encoding Synchronous Unison algorithm properties


---

## 📚 References

📄 A. Khoualdia, S. Cherif, S. Devismes, L. Robert. Analyzing Self-Stabilization of Synchronous Unison via Propositional Satisfiability. International Conference on Principles and Practice of Constraint Programming (CP 2025), Glasgow, Scotland. DOI: https://doi.org/10.4230/LIPIcs.CP.2025.19/

📄 Asma Khoualdia, Sami Cherif, Stéphane Devismes, Léo Robert. Analyse de l'autostabilisation de l'unisson synchrone via la satisfiabilité propositionnelle. Journées Francophones de Programmation par Contraintes (JFPC 2025), Jun 2025, Dijon, France. https://hal.science/hal-05208079/

---


## 📦 Installation & Usage (Generate a Single CNF Instance)

```bash
git clone <repository-url>
pip install python-sat[pblib,aiger]
python3 GraphSolver.py <graph_type> <num_nodes> <modulus> <CONV|DIV> <model>


