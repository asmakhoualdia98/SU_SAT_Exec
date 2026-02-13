def create_torus_adjacency_matrix(c):
    num_nodes = c * c  # Nombre total de nœuds
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        row, col = divmod(i, c)  # Convertir l'index en (ligne, colonne)
        
        # Voisin de droite (modulo pour connecter le bord droit avec le bord gauche)
        right = row * c + (col + 1) % c
        adjacency_matrix[i][right] = 1
        adjacency_matrix[right][i] = 1
        
        # Voisin de gauche (modulo pour connecter le bord gauche avec le bord droit)
        left = row * c + (col - 1) % c
        adjacency_matrix[i][left] = 1
        adjacency_matrix[left][i] = 1
        
        # Voisin du bas (modulo pour connecter le bas avec le haut)
        down = ((row + 1) % c) * c + col
        adjacency_matrix[i][down] = 1
        adjacency_matrix[down][i] = 1
        
        # Voisin du haut (modulo pour connecter le haut avec le bas)
        up = ((row - 1) % c) * c + col
        adjacency_matrix[i][up] = 1
        adjacency_matrix[up][i] = 1
    
    return adjacency_matrix
