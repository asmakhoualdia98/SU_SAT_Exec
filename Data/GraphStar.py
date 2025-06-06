def create_star_adjacency_matrix(num_nodes):
    # Initialize a square matrix of size num_nodes x num_nodes filled with zeros
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Loop through all nodes except the center (node 0)
    for i in range(1, num_nodes):
        # Connect the center node (0) to node i
        adjacency_matrix[0][i] = 1
        # Connect node i back to the center node (0), making the connection bidirectional
        adjacency_matrix[i][0] = 1

    # Return the adjacency matrix representing a star topology
    return adjacency_matrix
