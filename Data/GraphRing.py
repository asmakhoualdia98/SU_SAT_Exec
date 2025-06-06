def create_ring_adjacency_matrix(num_nodes):
    # Initialize a square matrix of size num_nodes x num_nodes filled with zeros
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Loop through each node to define its neighbors in the ring
    for i in range(num_nodes):
        # Calculate the index of the left neighbor (previous node), with wrap-around
        left = (i - 1) % num_nodes

        # Calculate the index of the right neighbor (next node), with wrap-around
        right = (i + 1) % num_nodes

        # Mark the connections in the adjacency matrix
        adjacency_matrix[i][left] = 1
        adjacency_matrix[i][right] = 1

    return adjacency_matrix
