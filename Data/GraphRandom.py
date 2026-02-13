import random
import math
import os



def create_erdos_renyi_adjacency_matrix(num_nodes):
    p = math.log(num_nodes) / num_nodes
    while True:
        adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < p:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
        
        # Vérification de connectivité intégrée
        n = len(adjacency_matrix)
        visited = [False] * n
        stack = [0]
        visited[0] = True

        while stack:
            node = stack.pop()
            for neighbor, edge in enumerate(adjacency_matrix[node]):
                if edge == 1 and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        if all(visited):
            return adjacency_matrix