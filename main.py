import numpy as np
import random
import itertools

# Função para calcular a distância total de uma rota
def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    num_locations = len(route)
    for i in range(num_locations - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]]  # Retorno ao ponto de partida
    return total_distance

# Função para criar uma rota aleatória
def create_route(num_locations):
    route = list(range(num_locations))
    random.shuffle(route)
    return route

# Função para criar uma população inicial de rotas
def create_initial_population(pop_size, num_locations):
    population = []
    for _ in range(pop_size):
        population.append(create_route(num_locations))
    return population

# Função para selecionar pais para crossover
def select_parents(population, fitness, num_parents):
    num_parents = min(num_parents, len(population))
    selected_indices = np.random.choice(len(population), size=num_parents, replace=False, p=fitness)
    selected_parents = [population[i] for i in selected_indices]
    return selected_parents

# Função para realizar o crossover entre dois pais
def crossover(parent1, parent2):
    child = [-1] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]
    
    fill_pos = end
    for gene in parent2:
        if gene not in child:
            if fill_pos == len(child):
                fill_pos = 0
            child[fill_pos] = gene
            fill_pos += 1
    return child

# Função para realizar mutação em uma rota
def mutate(route, mutation_rate):
    for swapped in range(len(route)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(route))
            route[swapped], route[swap_with] = route[swap_with], route[swapped]
    return route

# Função principal do Algoritmo Genético
def genetic_algorithm(distance_matrix, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500):
    num_locations = len(distance_matrix)
    population = create_initial_population(pop_size, num_locations)
    
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
            child = crossover(parent1, parent2)
            new_population.append(mutate(child, mutation_rate))
        
        population = new_population
    
    # Encontra a melhor rota e a distância mínima
    best_route = min(population, key=lambda route: calculate_total_distance(route, distance_matrix))
    best_distance = calculate_total_distance(best_route, distance_matrix)
    
    return best_route, best_distance

# Exemplo de uso com pontos turísticos
if __name__ == "__main__":
    # Matriz de distâncias entre pontos turísticos (exemplo)
    distance_matrix = np.array([
        [0, 29, 20, 21],  # Ponto Turístico A
        [29, 0, 15, 17],  # Ponto Turístico B
        [20, 15, 0, 28],  # Ponto Turístico C
        [21, 17, 28, 0]   # Ponto Turístico D
    ])
    
    # Nomes dos pontos turísticos correspondentes à matriz de distâncias
    tourist_spots = ["Ponto Turístico A", "Ponto Turístico B", "Ponto Turístico C", "Ponto Turístico D"]

    best_route, best_distance = genetic_algorithm(distance_matrix, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)
    
    # Traduzindo a rota em nomes de pontos turísticos
    best_route_named = [tourist_spots[i] for i in best_route]
    
    print("Melhor rota:", best_route_named)
    print("Distância mínima:", best_distance)
