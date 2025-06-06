# GraphSolver.py

import sys
import os
from GraphModel import GraphModel

VALID_GRAPH_TYPES = {"ring", "chain", "star"}
VALID_BEHAVIORS = {"CONV", "DIV"}
VALID_MODELS = {
    "INI", "ER", "OL", "ICP", "ICT",
    "ER-ICP", "ER-ICT", "OL-ICP", "OL-ICT"
}

def check_args(graph_type, num_nodes, modulus, behavior, model):
    errors = []

    if graph_type not in VALID_GRAPH_TYPES:
        errors.append(f"❌ Invalid topology: '{graph_type}'. Choose among {', '.join(VALID_GRAPH_TYPES)}.")

    if behavior not in VALID_BEHAVIORS:
        errors.append(f"❌ Invalid behavior: '{behavior}'. Choose 'CONV' or 'DIV'.")

    if model not in VALID_MODELS:
        errors.append(f"❌ Invalid configuration model: '{model}'. Choose among {', '.join(VALID_MODELS)}.")

    if behavior == "DIV" and any(opt in model for opt in ["ICP", "ICT", "ER-ICP", "ER-ICT", "OL-ICP", "OL-ICT"]):
        errors.append("❌ ICP or ICT models (or their combinations) are not compatible with DIV.")

    return errors

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("❗ Usage: python GraphSolver.py <graph_type> <num_nodes> <modulus> <CONV|DIV> <INI|ER|OL|...>")
        sys.exit(1)

    graph_type = sys.argv[1].lower()
    try:
        num_nodes = int(sys.argv[2])
        modulus = int(sys.argv[3])
    except ValueError:
        print("❗ 'num_nodes' and 'modulus' must be integers.")
        sys.exit(1)

    behavior = sys.argv[4].upper()
    model = sys.argv[5].upper()

    errors = check_args(graph_type, num_nodes, modulus, behavior, model)
    if errors:
        for error in errors:
            print(error)
        sys.exit(1)

    # Build the output path according to the requested structure
    output_dir = os.path.join("Benchmark", graph_type.capitalize(), behavior, model)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{graph_type}_{num_nodes}_{modulus}_{behavior}_{model}.cnf")

    try:
        model_instance = GraphModel(graph_type, num_nodes, modulus, behavior, model)
        model_instance.generate_cnf(output_path)  # call with single argument, output_path
        print(f"✅ CNF file generated: {output_path}")
    except Exception as e:
        print(f"❌ Error during model generation: {e}")
        sys.exit(1)
