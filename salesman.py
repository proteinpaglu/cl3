import random

# Distance between 4 cities (symmetric matrix)
distance = [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
]

num_cities = 4
num_ants = 4
num_iterations = 50
pheromone = [[1 for _ in range(num_cities)] for _ in range(num_cities)]
alpha = 1   # pheromone factor
beta = 2    # distance factor
evaporation = 0.5
Q = 100

def calculate_prob(current, visited):
    probs = []
    total = 0
    for j in range(num_cities):
        if j not in visited:
            tau = pheromone[current][j] ** alpha
            eta = (1 / distance[current][j]) ** beta
            total += tau * eta
            probs.append((j, tau * eta))
    return [(city, prob / total) for city, prob in probs]

def choose_next(probabilities):
    cities = [city for city, _ in probabilities]
    weights = [prob for _, prob in probabilities]
    return random.choices(cities, weights=weights)[0]

best_path = []
best_cost = float('inf')

for iteration in range(num_iterations):
    all_paths = []
    all_costs = []

    for ant in range(num_ants):
        path = []
        visited = set()
        current = random.randint(0, num_cities - 1)
        path.append(current)
        visited.add(current)

        while len(path) < num_cities:
            probs = calculate_prob(current, visited)
            next_city = choose_next(probs)
            path.append(next_city)
            visited.add(next_city)
            current = next_city

        path.append(path[0])  # return to start
        cost = sum(distance[path[i]][path[i+1]] for i in range(num_cities))
        all_paths.append(path)
        all_costs.append(cost)

        if cost < best_cost:
            best_cost = cost
            best_path = path

    # Update pheromones
    for i in range(num_cities):
        for j in range(num_cities):
            pheromone[i][j] *= (1 - evaporation)

    for path, cost in zip(all_paths, all_costs):
        for i in range(num_cities):
            a, b = path[i], path[i+1]
            pheromone[a][b] += Q / cost
            pheromone[b][a] += Q / cost

print("Best path:", best_path)
print("Shortest distance:", best_cost)
