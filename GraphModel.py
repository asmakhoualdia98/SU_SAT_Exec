# Standard libraries
import os    # For file system operations (e.g., creating directories)
import math  # For mathematical utilities (e.g., square root)
import json
from collections import deque

# PySAT library (used for CNF encoding and SAT problem formulation)
from pysat.formula import CNF              # To build and store CNF formulas
from pysat.card import CardEnc, EncType    # To encode cardinality constraints (e.g., at-most-one)

# Graph topology generators for different configurations
from Data.GraphRing import create_ring_adjacency_matrix   # Builds adjacency matrix for ring graphs
from Data.GraphChain import create_chain_adjacency_matrix # Builds adjacency matrix for chain graphs
from Data.GraphStar import create_star_adjacency_matrix   # Builds adjacency matrix for star graphs
from Data.GraphGrid import create_grid_adjacency_matrix   # Builds adjacency matrix for grid graphs    
from Data.GraphTorus import create_torus_adjacency_matrix # Builds adjacency matrix for torus graphs
from Data.GraphRandom import create_erdos_renyi_adjacency_matrix # Builds adjacency matrix for random graphs

class GraphModel:
    def __init__(self, graph_type, num_nodes, modulus, mode, model_option):
        # Type of the graph topology: "ring", "chain", "star", "grid" "torus", or "random"
        self.graph_type = graph_type
    
        # Number of processes (nodes) in the graph
        self.num_nodes = num_nodes
    
        # Modulus used for the clock values (defines clock domain: M = {0, ..., modulus - 1})
        self.modulus = modulus
        
    
        # Mode of the SAT model: "CONV" (convergence) or "DIV" (divergence)
        self.mode = mode.upper()
    
        # Optional optimizations or modeling flags (e.g., "ER", "LO", "ICT", "ICX")
        self.model_option = model_option.upper()
    
        # Square root of num_nodes (used for optional structures or heuristics)
        self.c = int(math.sqrt(num_nodes))
    
        # Compute the maximum number of synchronous steps to model (t_f)
        self.max_steps = self.calculate_max_steps()
    
    def calculate_diameter_random(self, adjacency_matrix):
        n = len(adjacency_matrix)
        def bfs(start):
            distances = [-1] * n
            distances[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor, edge in enumerate(adjacency_matrix[node]):
                    if edge == 1 and distances[neighbor] == -1:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            return max(distances)
        
        diameter = 0
        for node in range(n):
            diameter = max(diameter, bfs(node))
        return diameter
    
    def calculate_max_steps(self):
        # Computes the graph diameter D, which bounds how long clocks need to stabilize.

        if self.graph_type == "ring":
            D = self.num_nodes // 2 if self.num_nodes % 2 == 0 else (self.num_nodes - 1) // 2
        elif self.graph_type == "chain":
            D = self.num_nodes - 1
        elif self.graph_type == "star":
            D = 2  # Max distance in a star is always 2
        elif self.graph_type == "grid":
            D = 2 * ( self.c - 1)
        elif self.graph_type == "torus":
            D = 2 * (self.c // 2) if self.c % 2 == 0 else 2 * ((self.c - 1) // 2)
        elif self.graph_type == "random":
            D = self.calculate_diameter_random(create_erdos_renyi_adjacency_matrix(self.num_nodes))
            
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        return 5 * D
    
    
    def clock_var(self, i, t, value):
        return i * self.max_steps * self.modulus + t * self.modulus + value + 1

    def l_var(self, t):
       return self.clock_var(self.num_nodes - 1, self.max_steps - 1, self.modulus - 1) + t + 1

    def l_prime_var(self, i, t, value):
        return self.l_var(self.max_steps - 1) + i * self.max_steps * self.modulus + t * self.modulus + value + 1
        
    def n_var(self, i, t, value):
        return self.l_prime_var(self.num_nodes - 1, self.max_steps - 1, self.modulus - 1) + i * self.max_steps * self.modulus + t * self.modulus + value + 1


    def m_var(self, i, t, value):
        return self.n_var(self.num_nodes - 1, self.max_steps - 1, self.modulus - 1) + i * self.max_steps * self.modulus + t * self.modulus + value + 1
        
    def s_shift_var(self, i, j, t, value):
        return self.m_var(self.num_nodes - 1, self.max_steps - 1, self.modulus - 1) + i * self.num_nodes * self.max_steps * self.modulus + j * self.max_steps * self.modulus + t * self.modulus + value + 1
    
    def c_var(self, i, t):
        return self.s_shift_var(self.num_nodes - 1, self.num_nodes - 1, self.max_steps - 1, self.modulus - 1) + i * self.max_steps + t + 1
        
    
    
    
    def get_adjacency_matrix(self):
        # Returns the adjacency matrix corresponding to the selected graph type
        if self.graph_type == "ring":
            return create_ring_adjacency_matrix(self.num_nodes)
        elif self.graph_type == "chain":
            return create_chain_adjacency_matrix(self.num_nodes)
        elif self.graph_type == "star":
            return create_star_adjacency_matrix(self.num_nodes)
        elif self.graph_type == "grid":
            return create_grid_adjacency_matrix(self.num_nodes)
        elif self.graph_type == "torus":
            return create_torus_adjacency_matrix(self.num_nodes)
        elif self.graph_type == "random":
            return create_erdos_renyi_adjacency_matrix(self.num_nodes)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")


    def add_uniqueness_constraints(self, cnf):
        top_id = self.c_var(self.num_nodes - 1, self.max_steps - 1)
        for t in range(self.max_steps):
            for i in range(self.num_nodes):
                variables = [self.clock_var(i, t, v) for v in range(self.modulus)]
                
                card = CardEnc.equals(lits=variables, bound=1, encoding=EncType.cardnetwrk, top_id=top_id)
                
                top_id = card.nv
                cnf.extend(card.clauses)


    def add_update_clauses(self, cnf, adjacency_matrix):
        for t in range(self.max_steps - 1):
            for i in range(self.num_nodes):
                
                neighbors = [j for j in range(self.num_nodes) if adjacency_matrix[i][j] == 1]
                # For each value of the node at step t
                for current_value in range(self.modulus):  
                    for neighbor_values in range(self.modulus ** len(neighbors)):
                        
                        neighbor_state = [
                            (neighbor_values // (self.modulus ** k)) % self.modulus for k in range(len(neighbors))
                        ]
                        
                        
                        next_value = (min(neighbor_state + [current_value]) + 1) % self.modulus
                        
                        
                        clause = [-self.clock_var(i, t, current_value)]  
                        clause += [-self.clock_var(neighbors[k], t, neighbor_state[k]) for k in range(len(neighbors))]  
                        clause += [self.clock_var(i, t + 1, next_value)]  
                        
                        
                        cnf.append(clause)
                        

    def add_refined_update_clauses(self, cnf, adjacency_matrix):
        
        
        # ---------------------------------------------------------
        for t in range(self.max_steps - 1):  
            for p in range(self.num_nodes):
                # (N(p) ∪ {p})
                neighbors = [j for j in range(self.num_nodes) if adjacency_matrix[p][j] == 1] + [p]
                for v in range(self.modulus):
                    for p_prime in neighbors:
                        clause = [-self.clock_var(p_prime, t, v), -self.n_var(p, t, v)]
                        cnf.append(clause)
                        
                        
        for t in range(self.max_steps - 1):  
            for p in range(self.num_nodes):
                #  (N(p) ∪ {p})
                neighbors = [j for j in range(self.num_nodes) if adjacency_matrix[p][j] == 1] + [p]
                for v in range(self.modulus):
                    clause = [self.n_var(p, t, v)] + [self.clock_var(p_prime, t, v) for p_prime in neighbors]
                    cnf.append(clause)
    
        
        # ---------------------------------------------------------
        for t in range(self.max_steps - 1):  
            for p in range(self.num_nodes):
                # v=0
                cnf.append([-self.n_var(p, t, 0), self.m_var(p, t, 0)])
                # v > 0
                for v in range(1, self.modulus - 1):
                    clause = [
                        -self.n_var(p, t, v),
                        -self.m_var(p, t, v - 1),
                        self.m_var(p, t, v)
                    ]
                    cnf.append(clause)
                    
                    
        for t in range(self.max_steps - 1):  
            for p in range(self.num_nodes):
                # v=0
                cnf.append([self.n_var(p, t, 0), -self.m_var(p, t, 0)])
                # v > 0
                for v in range(1, self.modulus - 1):
                    clause1 = [
                        self.n_var(p, t, v),
                        -self.m_var(p, t, v)
                    ]
                    cnf.append(clause1)
                    clause2 = [
                        self.m_var(p, t, v - 1),
                        -self.m_var(p, t, v)
                    ]
                    cnf.append(clause2)
    
        
        # ---------------------------------------------------------
        for t in range(self.max_steps - 1):  # car on regarde h_{t+1}
            for p in range(self.num_nodes):
                clause1 = [
                    self.m_var(p, t, 0),
                    self.clock_var(p, t + 1, 1)
                ]
                cnf.append(clause1)
                
                for v in range(self.modulus - 2):
                    clause = [
                        -self.m_var(p, t, v),
                        self.m_var(p, t, v + 1),
                        self.clock_var(p, t + 1, v + 2)
                    ]
                    cnf.append(clause)
    
                
                clause = [
                    -self.m_var(p, t, self.modulus - 2),
                    self.clock_var(p, t + 1, 0)
                ]
                cnf.append(clause)


    def add_non_convergence_clauses(self, cnf):
        target_time = self.max_steps - 1
        for v in range(self.modulus):
            cnf.append([-self.clock_var(i, target_time, v) for i in range(self.num_nodes)])


    def generer_ensemble(self):
        T = {0}
        i = self.max_steps - 1 - self.modulus
        while i > 0:
            T.add(i)
            i -= self.modulus
        return T

    def add_non_convergence_opt_IC_P_clauses(self, cnf):
        T = self.generer_ensemble()
        for t in T:
            for v in range(self.modulus):
                cnf.append([-self.clock_var(i, t, v) for i in range(self.num_nodes)])
    
     

    def add_non_convergence_opt_IC_T_clauses(self, cnf):
        target_time = self.max_steps - 1
        for t in range(target_time):
            for v in range(self.modulus):
                cnf.append([-self.clock_var(i, t, v) for i in range(self.num_nodes)])
                
    
    
    def add_opt_er_torus_constraint(self, cnf):
        # --- Define C_i sets ---
        def circle_indices(i):
            return [(i * self.c + q) % self.num_nodes for q in range(self.c)]
    
        # (1) ¬p_{i,v} ∨ ¬h_{p,0,v'}
        for i in range(self.c):
            for v in range(1, self.modulus):
                for vp in range(v):
                    for p in circle_indices(i):
                        cnf.append([-self.p_var(i, v), -self.clock_var(p, 0, vp)])
    
        # (2) ∨_{v∈M} p_{i,v}
        for i in range(self.c):
            cnf.append([self.p_var(i, v) for v in range(self.modulus)])
    
        # (3) ¬p_{0,v} ∨ ¬p_{i,v'} for i != 0
        for i in range(1, self.c):
            for v in range(1, self.modulus):
                for vp in range(v):
                    cnf.append([-self.p_var(0, v), -self.p_var(i, vp)])
    

    
            
    def add_opt_er_ring_constraint(self, cnf):
        for v in range(self.modulus):  
            for u in range(v + 1, self.modulus):  
                for i in range(1, self.num_nodes):  
                    clause = [
                        -self.clock_var(i, 0, v),
                        -self.clock_var(0, 0, u)
                    ]
                    cnf.append(clause)
                    
    def add_opt_er_star_constraint(self, cnf):
        for v in range(self.modulus):  
            for u in range(v + 1, self.modulus):  
                for i in range(2, self.num_nodes):  
                    clause = [
                        -self.clock_var(i, 0, v),
                        -self.clock_var(1, 0, u)
                    ]
                    cnf.append(clause)
                    
    
    def add_opt_ol_star_clauses(self, cnf, adjacency_matrix):
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                
                neighbors_i = [k for k in range(self.num_nodes) if adjacency_matrix[i][k] == 1]
                neighbors_j = [k for k in range(self.num_nodes) if adjacency_matrix[j][k] == 1]
                if set(neighbors_i) == set(neighbors_j):
                    for v in range(self.modulus):
                        for v_prime in range(v + 1, self.modulus):
                            clause = [
                                -self.clock_var(i, 0, v),
                                -self.clock_var(j, 0, v_prime)
                            ]
                            cnf.append(clause)
                    
                    
    def add_cycle_uniqueness_constraints(self, cnf):
        # Start after all variables already used (including aux vars)
        top_id = cnf.nv
    
        # Collect all l_var(t) you want to constrain
        variables = [self.l_var(t) for t in range(1, self.max_steps)]
    
        # Enforce at most one true among them
        card = CardEnc.atmost(
            lits=variables,
            bound=1,
            encoding=EncType.cardnetwrk,
            top_id = top_id
        )
    
        cnf.extend(card.clauses)


    def add_divergence(self, cnf):
        
        # Formule 1
        
        clause = [self.l_var(t) for t in range(1, self.max_steps)]
        cnf.append(clause)
            
        # Formule 2 :
        
        for t in range(1, self.max_steps):
            for i in range(self.num_nodes):
                clause = [self.l_prime_var(i, t, v) for v in range(self.modulus)] + [-self.l_var(t)]
                cnf.append(clause)
     
        
        # Formule 3 :
        for t in range(1, self.max_steps):
            for i in range(self.num_nodes):
                for v in range(self.modulus):
                
                    clause1 = [
                        self.clock_var(i, 0, v), 
                        -self.l_prime_var(i,t, v)
                    ]
                    
                    cnf.append(clause1)
                    
                    clause2 = [
                        self.clock_var(i, t, v), 
                        -self.l_prime_var(i,t, v)
                    ]
                    
                    cnf.append(clause2)
                    
        # Formule 4 :
        for v in range(self.modulus):
            cnf.append([
                -self.clock_var(i, 0, v) for i in range(self.num_nodes)
            ])
            
            
    def shift_proc(self, p, w):
        return (p + w) % self.num_nodes
    
    def add_refined_divergence_ring(self, cnf):
    
        for p in range(self.num_nodes):
            for v in range(self.modulus):
                clause = [
                    self.c_var(w, t)
                    for t in range(1, self.max_steps)
                    for w in range(self.num_nodes)
                ]
                cnf.append(clause)
        
        for t in range(1, self.max_steps):
            for v in range(self.modulus):
                clause = [
                    -self.clock_var(p, 0, v) 
                    for p in range(self.num_nodes)
                ]
                cnf.append(clause)
        
        for t in range(1, self.max_steps):
            for p in range(self.num_nodes):
                for w in range(self.num_nodes):
                    clause = (
                        [-self.c_var(w, t)] +
                        [
                            self.s_shift_var(p, self.shift_proc(p, w), t, v)
                            for v in range(self.modulus)
                        ]
                    )
                    cnf.append(clause)
        
        for t in range(1, self.max_steps):
            for v in range(self.modulus):
                for p in range(self.num_nodes):
                    for w in range(self.num_nodes):
                        p_shift = self.shift_proc(p, w)
                        s = self.s_shift_var(p, p_shift, t, v)
    
                        cnf.append([-s, self.clock_var(p, 0, v)])
    
                        # (¬s ∨ h_{p_shift,t,v})
                        cnf.append([-s, self.clock_var(p_shift, t, v)])
                        
                        
    def shift_proc_torus(self, p, w):
        return (p + w * self.c) % self.num_nodes
    
    def add_refined_divergence_torus(self, cnf):
    
        for p in range(self.num_nodes):
            for v in range(self.modulus):
                clause = [
                    self.c_var(w, t)
                    for t in range(1, self.max_steps)
                    for w in range(self.num_nodes)
                ]
                cnf.append(clause)
        
        for t in range(1, self.max_steps):
            for v in range(self.modulus):
                clause = [
                    -self.clock_var(p, 0, v) 
                    for p in range(self.num_nodes)
                ]
                cnf.append(clause)
        
        for t in range(1, self.max_steps):
            for p in range(self.num_nodes):
                for w in range(self.num_nodes):
                    clause = (
                        [-self.c_var(w, t)] +
                        [
                            self.s_shift_var(p, self.shift_proc_torus(p, w), t, v)
                            for v in range(self.modulus)
                        ]
                    )
                    cnf.append(clause)
        
        for t in range(1, self.max_steps):
            for v in range(self.modulus):
                for p in range(self.num_nodes):
                    for w in range(self.num_nodes):
                        p_shift = self.shift_proc_torus(p, w)
                        s = self.s_shift_var(p, p_shift, t, v)
    
                        cnf.append([-s, self.clock_var(p, 0, v)])
    
                        # (¬s ∨ h_{p_shift,t,v})
                        cnf.append([-s, self.clock_var(p_shift, t, v)])


    def generate_cnf(self, output_path):
        # Initialize the CNF object where all clauses will be accumulated
        cnf = CNF()
    
        # Get the graph structure for neighborhood-dependent clauses
        adj = self.get_adjacency_matrix()
    
        # Add uniqueness constraints: each process has exactly one clock value at each time
        self.add_uniqueness_constraints(cnf)
        
    
        # Depending on the analysis mode, add the corresponding core property
        if self.mode == "CONV":
            self.add_non_convergence_clauses(cnf)
            
            if "DIR" in self.model_option:
                self.add_update_clauses(cnf, adj)
                
            if "COM" in self.model_option:
                self.add_refined_update_clauses(cnf, adj)
                
            if "ICT" in self.model_option:
                self.add_refined_update_clauses(cnf, adj)
                self.add_non_convergence_opt_IC_P_clauses(cnf)
                
            if "ICX" in self.model_option:
                self.add_refined_update_clauses(cnf, adj)
                self.add_non_convergence_opt_IC_T_clauses(cnf)
                
            if "RE" in self.model_option:
                self.add_refined_update_clauses(cnf, adj)
                
                if self.graph_type == "ring":
                    self.add_opt_er_ring_constraint(cnf)
                    
                elif self.graph_type == "star":
                    self.add_opt_er_star_constraint(cnf)
                    
            if "LO" in self.model_option and self.graph_type == "star":
                self.add_refined_update_clauses(cnf, adj)
                self.add_opt_ol_star_clauses(cnf, adj)
    
        elif self.mode == "DIV":
        
            if "DIR" in self.model_option:
                self.add_update_clauses(cnf, adj)
                self.add_divergence(cnf)
                
            if "COM" in self.model_option:
                self.add_refined_update_clauses(cnf, adj)
                self.add_divergence(cnf)
                
            if "CU" in self.model_option:
                self.add_refined_update_clauses(cnf, adj)
                self.add_divergence(cnf)
                self.add_cycle_uniqueness_constraints(cnf)
                
            if "SC" in self.model_option:
                self.add_refined_update_clauses(cnf, adj)
                if self.graph_type == "ring":
                    self.add_refined_divergence_ring(cnf)  
                if self.graph_type == "torus":
                    self.add_refined_divergence_torus(cnf)
                
            
    
        # Optional optimizations based on structural symmetry
    
        
    
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
        # Write the final CNF clauses to the specified file in DIMACS format
        cnf.to_file(output_path)
        print(f"✅ CNF file generated: {output_path}")
