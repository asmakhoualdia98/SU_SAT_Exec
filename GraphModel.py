# Standard libraries
import os    # For file system operations (e.g., creating directories)
import math  # For mathematical utilities (e.g., square root)

# PySAT library (used for CNF encoding and SAT problem formulation)
from pysat.formula import CNF              # To build and store CNF formulas
from pysat.card import CardEnc, EncType    # To encode cardinality constraints (e.g., at-most-one)

# Graph topology generators for different configurations
from Data.GraphRing import create_ring_adjacency_matrix   # Builds adjacency matrix for ring graphs
from Data.GraphChain import create_chain_adjacency_matrix # Builds adjacency matrix for chain graphs
from Data.GraphStar import create_star_adjacency_matrix   # Builds adjacency matrix for star graphs


class GraphModel:
    def __init__(self, graph_type, num_nodes, modulus, mode, model_option):
        # Type of the graph topology: "ring", "chain", or "star"
        self.graph_type = graph_type
    
        # Number of processes (nodes) in the graph
        self.num_nodes = num_nodes
    
        # Modulus used for the clock values (defines clock domain: M = {0, ..., modulus - 1})
        self.modulus = modulus
    
        # Mode of the SAT model: "CONV" (convergence) or "DIV" (divergence)
        self.mode = mode.upper()
    
        # Optional optimizations or modeling flags (e.g., "ER", "OL", "ICP", "ICT")
        self.model_option = model_option.upper()
    
        # Square root of num_nodes (used for optional structures or heuristics)
        self.c = int(math.sqrt(num_nodes))
    
        # Compute the maximum number of synchronous steps to model (t_f)
        self.max_steps = self.calculate_max_steps()
    
    
    def calculate_max_steps(self):
        # Computes the graph diameter D, which bounds how long clocks need to stabilize.
        # Then, returns 3 * D as a safe upper bound for maximum execution steps (t_f).
        if self.graph_type == "ring":
            D = self.num_nodes // 2 if self.num_nodes % 2 == 0 else (self.num_nodes - 1) // 2
        elif self.graph_type == "chain":
            D = self.num_nodes - 1
        elif self.graph_type == "star":
            D = 2  # Max distance in a star is always 2
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        return 3 * D
    
    
    def clock_var(self, i, t, value):
        # Encodes the Boolean variable for: "Process i has clock value `value` at time `t`"
        # Unique variable index using offset: base + i·T·M + t·M + value + 1
        return i * self.max_steps * self.modulus + t * self.modulus + value + 1
    
    
    def l_var(self, t):
        # Boolean variable indicating that time `t` is part of a divergence cycle
        # Offset starts after all clock variables: num_nodes·T·M
        return self.num_nodes * self.max_steps * self.modulus + t + 1
    
    
    def l_prime_var(self, i, t, v):
        # Auxiliary variable used in divergence encoding:
        # True if process `i` has clock value `v` at time `t` AND time `t` is in a cycle
        # Offset continues after l_var (max_steps vars) + offset by process, time, and value
        return (self.num_nodes * self.max_steps * self.modulus) + self.max_steps + i * self.max_steps * self.modulus + t * self.modulus + v + 1
    
    
    def get_adjacency_matrix(self):
        # Returns the adjacency matrix corresponding to the selected graph type
        if self.graph_type == "ring":
            return create_ring_adjacency_matrix(self.num_nodes)
        elif self.graph_type == "chain":
            return create_chain_adjacency_matrix(self.num_nodes)
        elif self.graph_type == "star":
            return create_star_adjacency_matrix(self.num_nodes)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")


    def add_uniqueness_constraints(self, formula):
        # Compute the highest variable index currently in use (for CNF bookkeeping)
        # This ensures that the cardinality encoder generates new variable IDs safely
        top_id = self.clock_var(self.num_nodes - 1, self.max_steps - 1, self.modulus - 1)
    
        # Iterate over all time steps t ∈ T
        for t in range(self.max_steps):
    
            # Iterate over all processes p ∈ V (indexed by i)
            for i in range(self.num_nodes):
    
                # Build the list of all Boolean variables h_{p,t,v} for all v ∈ M
                # These variables represent that process p has clock value v at time t
                vars_ = [self.clock_var(i, t, v) for v in range(self.modulus)]
    
                # Enforce the constraint: ∑_{v ∈ M} h_{p,t,v} = 1
                # This is exactly constraint (1): each process has exactly one clock value at each time step
                card = CardEnc.equals(
                    lits=vars_,           # the literals h_{p,t,v}
                    bound=1,              # enforce exactly one to be true
                    encoding=EncType.cardnetwrk,  # use a network-based encoding for cardinality constraints
                    top_id=top_id         # ensure fresh variable IDs start above current max
                )
    
                # Update top_id to reflect the new highest variable ID used by the encoder
                top_id = card.nv
    
                # Add the generated CNF clauses to the global formula
                formula.extend(card.clauses)


    def add_update_clauses(self, cnf, adjacency_matrix):
        # Iterate over all time steps t ∈ T \ {t_f - 1}
        for t in range(self.max_steps - 1):
    
            # Iterate over all processes p ∈ V (indexed by i)
            for i in range(self.num_nodes):
    
                # Identify the neighbors N(p) of node i using the adjacency matrix
                neighbors = [j for j in range(self.num_nodes) if adjacency_matrix[i][j] == 1]
    
                # Loop over all possible values that process i can have at time t
                for current_value in range(self.modulus):  # current_value ∈ M
    
                    # Enumerate all possible combinations of values assigned to neighbors
                    # There are m^|N(p)| such combinations, where m = self.modulus
                    for neighbor_values in range(self.modulus ** len(neighbors)):
    
                        # Convert the integer neighbor_values to a list of individual values
                        # This gives one specific value assignment to all neighbors at time t
                        neighbor_state = [
                            (neighbor_values // (self.modulus ** k)) % self.modulus for k in range(len(neighbors))
                        ]
    
                        # Build the full closed neighborhood values: N(p) ∪ {p}
                        closed_neighborhood = neighbor_state + [current_value]
    
                        # Compute the next value for process i at time t+1
                        # According to the rule: v' = (min(v_p' for p' ∈ N(p) ∪ {p}) + 1) mod m
                        next_value = (min(closed_neighborhood) + 1) % self.modulus
    
                        # Build the clause:
                        # (¬h_{p,t,v}) ∨ (¬h_{n1,t,v1}) ∨ ... ∨ (¬h_{nd,t,vd}) ∨ h_{p,t+1,v'}
                        # This corresponds to:
                        #     (∨ ¬h_{p',t,v_{p'}} for p' ∈ N(p) ∪ {p}) ∨ h_{p,t+1,v'}
                        clause = [-self.clock_var(i, t, current_value)]  # ¬h_{p,t,v}
                        clause += [-self.clock_var(neighbors[k], t, neighbor_state[k]) for k in range(len(neighbors))]  # ¬h_{n_k,t,v_k}
                        clause += [self.clock_var(i, t + 1, next_value)]  # h_{p,t+1,v'}
    
                        # Add the clause to the CNF formula
                        # This encodes the clock update constraint in SAT form
                        cnf.append(clause)


    def add_non_convergence_clauses(self, cnf):
        # Define the target time t_f - 1 (i.e., the last configuration of the execution)
        target_time = self.max_steps - 1
    
        # For every possible clock value v ∈ M
        for v in range(self.modulus):
            # Build the clause: (∨_{p ∈ V} ¬h_{p,t_f−1,v})
            # Meaning: not all processes have value v at time t_f - 1 — i.e., configuration is not legitimate
            clause = [-self.clock_var(i, target_time, v) for i in range(self.num_nodes)]
    
            # Add this clause to the CNF formula
            # The full set of clauses encodes: ∧_{v ∈ M} ∨_{p ∈ V} ¬h_{p,t_f−1,v}
            cnf.append(clause)


    def add_divergence(self, cnf):
        # Enforce the existence of at least one cycle (i.e., one configuration t > 0 
        # such that configuration 0 and t are identical).
        # This corresponds to:  ∨_{t ∈ T, t > 0} c_t  (Eq. cyclediv1)
        cnf.append([self.l_var(t) for t in range(1, self.max_steps)])
    
        # For each time step t > 0 (candidate cycle point)
        for t in range(1, self.max_steps):
            for i in range(self.num_nodes):
                # For each process i, if a cycle exists at time t (i.e., l_t is true),
                # then there must exist at least one clock value v such that process i
                # has the same value v at both times 0 and t.
                # This encodes:  ¬c_t ∨ ∨_{v ∈ M} s_{p,t,v}  (Eq. cycle1)
                cnf.append([self.l_prime_var(i, t, v) for v in range(self.modulus)] + [-self.l_var(t)])
    
                for v in range(self.modulus):
                    # If s_{p,t,v} is true (i.e., process i has value v at both 0 and t),
                    # then it must be that h_{p,0,v} ∧ h_{p,t,v} is true.
                    # Which is rewritten in CNF as:
                    #    s_{p,t,v} → h_{p,0,v} ∧ h_{p,t,v}
                    # becomes:
                    #    ¬s_{p,t,v} ∨ h_{p,0,v}
                    #    ¬s_{p,t,v} ∨ h_{p,t,v}
                    cnf.append([self.clock_var(i, 0, v), -self.l_prime_var(i, t, v)])
                    cnf.append([self.clock_var(i, t, v), -self.l_prime_var(i, t, v)])
    
        # Finally, to ensure that the cycle is over **illegitimate configurations**, 
        # we encode that configuration 0 is illegitimate:
        # For all values v ∈ M, there exists at least one process p such that
        # h_{p,0,v} is false. This is:
        # ∧_{v ∈ M} ∨_{p ∈ V} ¬h_{p,0,v}  (Eq. cyclediv2)
        for v in range(self.modulus):
            cnf.append([-self.clock_var(i, 0, v) for i in range(self.num_nodes)])


    def add_opt_er_ring_constraint(self, cnf):
        # This constraint enforces rotation elimination (RE) for ring topologies.
        # It ensures that process 0 has the smallest clock value among all processes at t=0.
        # For all pairs (v, u) such that v < u, and for all i ≠ 0,
        # it adds the clause: ¬x_{i,0,v} ∨ ¬x_{0,0,u}
        # which means: if process i has a smaller value v, then process 0 cannot have a larger value u.
        # This removes redundant rotated initial configurations by keeping only canonical ones.
    
        for v in range(self.modulus):
            for u in range(v + 1, self.modulus):  # u > v
                for i in range(1, self.num_nodes):  # all processes except 0
                    cnf.append([
                        -self.clock_var(i, 0, v),   # ¬x_{i,0,v}
                        -self.clock_var(0, 0, u)    # ¬x_{0,0,u}
                    ])


    def add_opt_er_star_constraint(self, cnf):
        # This constraint is a variant of rotation elimination adapted for star topologies.
        # It ensures that process 1 has the smallest clock value among the peripheral processes at t=0.
        # For all pairs (v, u) such that v < u, and for i ≥ 2,
        # it adds the clause: ¬x_{i,0,v} ∨ ¬x_{1,0,u}
        # This avoids analyzing symmetric configurations where the roles of nodes with
        # the same neighborhood can be permuted without changing behavior.
    
        for v in range(self.modulus):
            for u in range(v + 1, self.modulus):  # u > v
                for i in range(2, self.num_nodes):  # only peripheral nodes excluding center (assumed 0 or 1)
                    cnf.append([
                        -self.clock_var(i, 0, v),   # ¬x_{i,0,v}
                        -self.clock_var(1, 0, u)    # ¬x_{1,0,u}
                    ])


    def add_opt_ol_star_clauses(self, cnf, adjacency_matrix):
        # For each pair of processes (i, j) such that i < j
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # Compute the neighbors of i and j
                neighbors_i = [k for k in range(self.num_nodes) if adjacency_matrix[i][k] == 1]
                neighbors_j = [k for k in range(self.num_nodes) if adjacency_matrix[j][k] == 1]
    
                # If processes i and j have the same neighborhood
                if set(neighbors_i) == set(neighbors_j):
                    # Impose lexicographic order on their initial clock values
                    # For all v, v' in M such that v' < v, we add the clause:
                    # ¬x_{i,0,v} ∨ ¬x_{j,0,v'}
                    # This eliminates initial configurations where process j has a smaller
                    # clock value than i when both have the same neighbors.
                    for v in range(self.modulus):
                        for v_prime in range(v + 1, self.modulus):
                            cnf.append([
                                -self.clock_var(i, 0, v),   # ¬x_{i,0,v}
                                -self.clock_var(j, 0, v_prime)  # ¬x_{j,0,v'}
                            ])


    def generer_ensemble(self):
        # This helper function generates a subset T of time steps where convergence checks will be added.
        # According to the IC_P strategy, it selects time steps spaced by the modulus (m), starting from t_f - 1 - m.
        # It ensures t = 0 is always included, representing the initial configuration.
        # This reduces the number of convergence constraints while still regularly checking legitimacy.
    
        T = {0}
        i = self.max_steps - 1 - self.modulus
        while i > 0:
            T.add(i)
            i -= self.modulus
        return T


    def add_non_convergence_opt_IC_P_clauses(self, cnf):
        # This function encodes the IC_P (Iterative Convergence on periodic subset) constraint.
        # It uses the set T generated by generer_ensemble(), which includes t=0 and every m-th step before t_f-1.
        # For each selected time t in T and for each clock value v,
        # it adds a clause: ∨_{p in V} ¬x_{p,t,v}
        # This clause ensures that the configuration at time t is not legitimate (not all clocks equal),
        # helping to prove non-convergence from at least one initial configuration.
    
        T = self.generer_ensemble()
        for t in T:
            for v in range(self.modulus):
                cnf.append([
                    -self.clock_var(i, t, v) for i in range(self.num_nodes)
                ])


    def add_non_convergence_opt_IC_T_clauses(self, cnf):
        # This function encodes the IC_T (Iterative Convergence over full execution) constraint.
        # It adds the same non-convergence clause at *every* time step t < t_f - 1.
        # These constraints enforce that no legitimate configuration appears before the final step,
        # strengthening the model and potentially speeding up detection of convergence or counterexamples.
    
        target_time = self.max_steps - 1
        for t in range(target_time):
            for v in range(self.modulus):
                cnf.append([
                    -self.clock_var(i, t, v) for i in range(self.num_nodes)
                ])


    def generate_cnf(self, output_path):
        # Initialize the CNF object where all clauses will be accumulated
        cnf = CNF()
    
        # Get the graph structure for neighborhood-dependent clauses
        adj = self.get_adjacency_matrix()
    
        # Add uniqueness constraints: each process has exactly one clock value at each time
        self.add_uniqueness_constraints(cnf)
    
        # Add transition/update clauses modeling the unison algorithm's rules
        self.add_update_clauses(cnf, adj)
    
        # Depending on the analysis mode, add the corresponding core property
        if self.mode == "CONV":
            # Add the main convergence constraint (¬legitimate at tf - 1)
            self.add_non_convergence_clauses(cnf)
    
            # Optionally enhance convergence checking with iterative constraints
            if "ICP" in self.model_option:
                # IC_P: non-convergence enforced periodically across execution
                self.add_non_convergence_opt_IC_P_clauses(cnf)
            if "ICT" in self.model_option:
                # IC_T: non-convergence enforced at every step before tf - 1
                self.add_non_convergence_opt_IC_T_clauses(cnf)
    
        elif self.mode == "DIV":
            # Add divergence modeling constraints based on illegitimate cycle detection
            self.add_divergence(cnf)
    
        # Optional optimizations based on structural symmetry
    
        if "ER" in self.model_option:
            # ER (Rotation Elimination) optimization to eliminate redundant initial configurations
            if self.graph_type == "ring":
                self.add_opt_er_ring_constraint(cnf)
            elif self.graph_type == "star":
                self.add_opt_er_star_constraint(cnf)
    
        if "OL" in self.model_option and self.graph_type == "star":
            # OL (Lexicographic Order) optimization for symmetric processes in star topologies
            self.add_opt_ol_star_clauses(cnf, adj)
    
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
        # Write the final CNF clauses to the specified file in DIMACS format
        cnf.to_file(output_path)
        print(f"✅ CNF file generated: {output_path}")
