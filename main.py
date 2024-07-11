import numpy as np
from utils import genetic_algorithm
import matplotlib.pyplot as plt
from utils import plot_route, plot_evolution, create_initial_population

# Definindo as localidades e os índices correspondentes
locations = [
    "Teatro Amazonas",
    "MUSA",
    "Encontro das Águas",
    "EST",
    "Ponta Negra"
]

coordinates = [
    (4, 1),
    (8, 8),
    (9, 2),
    (5, 3),
    (1, 5)
]

# Matriz de distâncias (km)
distance_matrix = np.array([
    [0, 19, 12, 7, 15],   # Teatro Amazonas
    [19, 0, 24, 16, 99999999],  # MUSA
    [12, 24, 0, 10, 99999999],  # Encontro das Águas
    [7, 16, 10, 0, 13],   # EST
    [15, 99999999, 99999999, 13, 0]  # Ponta Negra
])

# Matriz de custos (reais)
cost_matrix = np.array([
    [0, 35, 25, 10, 25],   # Teatro Amazonas
    [35, 0, 35, 30, 99999999],  # MUSA
    [25, 35, 0, 30, 99999999],  # Encontro das Águas
    [10, 30, 30, 0, 20],   # EST
    [25, 99999999, 99999999, 20, 0]  # Ponta Negra
])


def main():
    print("Bem-vindo ao sistema de rotas turísticas de Manaus!")

    print("Escolha uma das opções abaixo:")
    print("1. Caminho mais curto")
    print("2. Uber mais barato")

    option_selected = int(input("Digite a opção desejada: "))

    if option_selected == 1:
        matrix = distance_matrix
    else:
        matrix = cost_matrix

    print("Escolha uma das opções abaixo:")
    print("1. Escolher um ponto de início e fim")
    print("2. Escolher um ponto de início")
    print("3. Escolher um ponto de fim")
    print("4. Não escolher nem início nem fim")

    option = int(input("Digite a opção desejada: "))

    start, end = None, None

    if option == 1:
        print("Escolha um ponto de início:")
        for i, location in enumerate(locations):
            print(i, location)
        start = int(input("Digite o ponto de início: "))
        
        print("Escolha um ponto de fim:")
        for i, location in enumerate(locations):
            print(i, location)
        end = int(input("Digite o ponto de fim: "))

    elif option == 2:
        print("Escolha um ponto de início:")
        for i, location in enumerate(locations):
            print(i, location)
        start = int(input("Digite o ponto de início: "))

    elif option == 3:
        print("Escolha um ponto de fim:")
        for i, location in enumerate(locations):
            print(i, location)
        end = int(input("Digite o ponto de fim: "))

    print("Calculando a melhor rota...")
    best_route, best_distance, best_distances = genetic_algorithm(matrix, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500, start=start, end=end)

    best_route_named = [locations[i] for i in best_route]

    print("Melhor rota:", best_route_named)
    if option_selected == 1:
        print("Distância mínima:", best_distance, "km")
    else:
        print("Custo mínimo: R$", best_distance)

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    
    # Plotar a melhor rota
    plot_route(best_route, coordinates, locations, "Melhor Rota Encontrada", axs[0], color='blue')
    
    # Plotar a evolução da distância mínima
    plot_evolution(best_distances, axs[1])
    
    # Plotar algumas soluções iniciais aleatórias para comparação
    initial_routes = create_initial_population(5, len(coordinates), start=start, end=end)
    colors = ['gray', 'red', 'green', 'purple', 'orange']
    for i, route in enumerate(initial_routes):
        plot_route(route, coordinates, locations, f"Solução Inicial {i+1}", axs[2], color=colors[i])
    
    # Adicionar a melhor rota também no terceiro gráfico para comparação
    plot_route(best_route, coordinates, locations, "Comparação com Melhor Rota", axs[2], color='blue')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()