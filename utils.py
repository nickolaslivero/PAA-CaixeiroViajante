import numpy as np
import random
import matplotlib.pyplot as plt

# Função para calcular a distância total de uma rota
def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    num_locations = len(route)
    for i in range(num_locations - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]] if route[0] != route[-1] else 0  # Retorno ao ponto de partida se aplicável
    return total_distance

# Função para criar uma rota aleatória considerando os parâmetros start e end
def create_route(num_locations, start=None, end=None):
    route = list(range(num_locations))
    if start is not None:
        route.remove(start)
    if end is not None:
        route.remove(end)
    
    random.shuffle(route)
    
    if start is not None:
        route.insert(0, start)
    if end is not None:
        route.append(end)
        
    return route

# Função para criar uma população inicial de rotas
def create_initial_population(pop_size, num_locations, start=None, end=None):
    population = []
    for _ in range(pop_size):
        population.append(create_route(num_locations, start, end))
    return population

# Função para selecionar pais para crossover
def select_parents(population, fitness, num_parents):
    num_parents = min(num_parents, len(population))
    selected_indices = np.random.choice(len(population), size=num_parents, replace=False, p=fitness)
    selected_parents = [population[i] for i in selected_indices]
    return selected_parents

# Função para realizar o crossover entre dois pais
def crossover(parent1, parent2, start=None, end=None):
    child = [-1] * len(parent1)
    if start is not None:
        child[0] = start
    if end is not None:
        child[-1] = end
    
    parent1_mid = parent1[1:-1] if start is not None and end is not None else parent1[1:] if start is not None else parent1[:-1] if end is not None else parent1
    parent2_mid = parent2[1:-1] if start is not None and end is not None else parent2[1:] if start is not None else parent2[:-1] if end is not None else parent2
    mid_start, mid_end = sorted(random.sample(range(len(parent1_mid)), 2))
    
    if start is not None:
        child[mid_start + 1: mid_end + 1] = parent1_mid[mid_start: mid_end]
    else:
        child[mid_start: mid_end] = parent1_mid[mid_start: mid_end]
    
    fill_pos = mid_end + 1 if start is not None else mid_end
    for gene in parent2_mid:
        if gene not in child:
            if fill_pos == len(child):
                fill_pos = 1 if start is not None else 0
            while child[fill_pos] != -1:
                fill_pos += 1
                if fill_pos == len(child):
                    fill_pos = 1 if start is not None else 0
            child[fill_pos] = gene
    return child

# Função para realizar mutação em uma rota
def mutate(route, mutation_rate, start=None, end=None):
    for swapped in range(1 if start is not None else 0, len(route) - 1 if end is not None else len(route)):
        if random.random() < mutation_rate:
            swap_with = random.randint(1 if start is not None else 0, len(route) - 2 if end is not None else len(route) - 1)
            route[swapped], route[swap_with] = route[swap_with], route[swapped]
    return route

# Função principal do Algoritmo Genético
def genetic_algorithm(distance_matrix, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500, start=None, end=None):
    num_locations = len(distance_matrix)
    population = create_initial_population(pop_size, num_locations, start, end)
    best_distances = []

    for generation in range(generations):
        # Avalia a aptidão de cada rota com base na distância total
        fitness = np.array([1 / calculate_total_distance(route, distance_matrix) for route in population])
        fitness = fitness / fitness.sum()
        
        new_population = []
        # Preserva os melhores indivíduos (elite)
        elite_indices = np.argsort(fitness)[-elite_size:]
        elite = [population[i] for i in elite_indices]
        new_population.extend(elite)
        
        # Seleciona os pais para a reprodução
        num_parents = pop_size - elite_size
        parents = select_parents(population, fitness, num_parents)
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i+1) % len(parents)]
            child = crossover(parent1, parent2, start, end)
            new_population.append(mutate(child, mutation_rate, start, end))
        
        population = new_population
        
        # Armazena a melhor distância da geração
        best_distance = calculate_total_distance(min(population, key=lambda route: calculate_total_distance(route, distance_matrix)), distance_matrix)
        best_distances.append(best_distance)
    
    # Encontra a melhor rota e a distância mínima
    best_route = min(population, key=lambda route: calculate_total_distance(route, distance_matrix))
    best_distance = calculate_total_distance(best_route, distance_matrix)
    
    return best_route, best_distance, best_distances

# Função para plotar a rota
def plot_route(route, coordinates, tourist_spots, title, ax, color='blue'):
    route_points = [coordinates[i] for i in route]
    
    x = [point[0] for point in route_points]
    y = [point[1] for point in route_points]
    ax.plot(x, y, 'o-', color=color)
    
    for i, spot in enumerate(route_points):
        ax.annotate(tourist_spots[route[i]], (spot[0], spot[1]))
    
    ax.set_title(title)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")

# Função para plotar a evolução da distância mínima
def plot_evolution(best_distances, ax):
    ax.plot(best_distances)
    ax.set_title('Evolução da Distância Mínima')
    ax.set_xlabel('Gerações')
    ax.set_ylabel('Distância Mínima')

# Função para calcular a matriz de distâncias
def calculate_distance_matrix(coordinates):
    num_locations = len(coordinates)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i][j] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
    return distance_matrix

# Exemplo de uso com pontos turísticos
if __name__ == "__main__":
    # Coordenadas dos pontos turísticos (exemplo)
    coordinates = [
        (0, 0), (1, 3), (4, 3), (6, 1), (3, 0),
        (5, 5), (8, 8), (7, 2)
    ]
    
    # Calcular a matriz de distâncias com base nas coordenadas
    distance_matrix = calculate_distance_matrix(coordinates)

    print(distance_matrix)
    
    # Nomes dos pontos turísticos correspondentes às coordenadas
    tourist_spots = ["Ponto A", "Ponto B", "Ponto C", "Ponto D", "Ponto E", "Ponto F", "Ponto G", "Ponto H"]

    start, end = None, None  # Definir os pontos de início e fim (exemplo)
    best_route, best_distance, best_distances = genetic_algorithm(distance_matrix, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500, start=start, end=end)
    
    # Traduzindo a rota em nomes de pontos turísticos
    best_route_named = [tourist_spots[i] for i in best_route]
    
    print("Melhor rota:", best_route_named)
    print("Distância mínima:", best_distance)
    
    # Criar a figura com subplots
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    
    # Plotar a melhor rota
    plot_route(best_route, coordinates, tourist_spots, "Melhor Rota Encontrada", axs[0], color='blue')
    
    # Plotar a evolução da distância mínima
    plot_evolution(best_distances, axs[1])
    
    # Plotar algumas soluções iniciais aleatórias para comparação
    initial_routes = create_initial_population(5, len(coordinates), start=start, end=end)
    colors = ['gray', 'red', 'green', 'purple', 'orange']
    for i, route in enumerate(initial_routes):
        plot_route(route, coordinates, tourist_spots, f"Solução Inicial {i+1}", axs[2], color=colors[i])
    
    # Adicionar a melhor rota também no terceiro gráfico para comparação
    plot_route(best_route, coordinates, tourist_spots, "Comparação com Melhor Rota", axs[2], color='blue')
    
    plt.tight_layout()
    plt.show()
