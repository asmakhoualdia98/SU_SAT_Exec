# Analyzing Synchronous Unison Through SAT

## 🧩 Overview

This project is a Python-based framework for modeling and analyzing **self-stabilizing algorithms**, specifically targeting the **Synchronous Unison algorithm** by *Arora et al.* The framework generates corresponding CNF (Conjunctive Normal Form) files to be used with SAT solvers for formal verification and analysis of this algorithm.

The project allows simulation over various graph topologies, configuration models, and behavioral assumptions to assess algorithm correctness under different structural constraints.

The project includes tools for:

- CNF file generation from graph and algorithm parameters
- Batch instance creation for broad exploration
- SAT solver execution for scalability
- Hierarchical benchmark organization for structured experimentation

---

## 📚 References
📄 Analyzing Self-Stabilization of Synchronous Unison via Propositional Satisfiability
A. Khoualdia, S. Cherif, S. Devismes, L. Robert, CP 2025 (International Conference on Principles and Practice of Constraint Programming), Glasgow, Scotland.

🧪 Analyzing Self-Stabilization of Synchronous Unison via Propositional Satisfiability
A. Khoualdia, S. Cherif, S. Devismes, L. Robert, JFPC 2025 (Journées Francophones de Programmation par Contraintes), Dijon, France.

---

## 🚀 Features

- ✅ Supports multiple graph types: `ring`, `chain`, `star`
- 🔁 Behavior simulation: `CONV` (converging) and `DIV` (diverging)
- ⚙️ Model options: `INI`, `ER`, `OL`, `ICP`, `ICT`, `ER-ICP`, `ER-ICT`, `OL-ICP`, `OL-ICT`
- 🛠 Generates CNF files encoding Synchronous Unison algorithm properties
- 🧠  Batch generation and SAT solving
- 📁 Auto-organized benchmark directory structure

---

## 📦 Installation & Usage (Generate a Single CNF Instance)

```bash
git clone <repository-url>
cd CP_Code_Parallel_SAT_SU_Exec
pip install python-sat[pblib,aiger]
python3 GraphSolver.py <graph_type> <num_nodes> <modulus> <CONV|DIV> <model>
