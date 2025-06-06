def create_chain_adjacency_matrix(num_nodes):
    # Initialize a square matrix of size num_nodes x num_nodes filled with zeros
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Fill the matrix to reflect the chain structure
    for i in range(num_nodes):
        if i > 0:
            # Connect node i to its previous node (i-1)
            adjacency_matrix[i][i - 1] = 1
            adjacency_matrix[i - 1][i] = 1  # Ensure the connection is bidirectional (undirected graph)

    return adjacency_matrix
