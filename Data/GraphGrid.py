def create_grid_adjacency_matrix(c):
    num_nodes = c * c  # Nombre total de nœuds
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        row, col = divmod(i, c)  # Convertir l'index en (ligne, colonne)
        
        # Ajouter les connexions aux voisins
        if col < c - 1:  # Voisin de droite
            adjacency_matrix[i][i + 1] = 1
            adjacency_matrix[i + 1][i] = 1
        
        if row < c - 1:  # Voisin du bas
            adjacency_matrix[i][i + c] = 1
            adjacency_matrix[i + c][i] = 1
    
    return adjacency_matrix