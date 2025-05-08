import numpy as np
import random

# Rastrigin Function (Optimization Objective)
def rastrigin(x):
    A = 10
    return A * len(x) + sum(x_i**2 - A * np.cos(2 * np.pi * x_i) for x_i in x)

# Clonal Selection Algorithm
def clonal_selection_algorithm(pop_size, generations, mutation_rate, elite_size):
    # Step 1: Initialize population with random values between -5.12 and 5.12
    population = np.random.uniform(-5.12, 5.12, (pop_size, 2))  # 2D Rastrigin Problem
    
    # Step 2: Start evolution for 'generations' number of generations
    for gen in range(generations):
        # Step 3: Calculate the fitness for each individual in the population
        fitness = np.array([rastrigin(ind) for ind in population])

        # Step 4: Select the best (elite) individuals based on their fitness
        elite_indices = np.argsort(fitness)[:elite_size]  # Get the indices of the elite individuals
        elite_individuals = population[elite_indices]

        # Step 5: Clone the best individuals
        clones = elite_individuals.copy()

        # Step 6: Mutate the clones slightly (explore new solutions)
        for i in range(len(clones)):
            if random.random() < mutation_rate:
                clones[i] += np.random.uniform(-0.1, 0.1, 2)  # Small random change
                clones[i] = np.clip(clones[i], -5.12, 5.12)  # Ensure values are within the bounds

        # Step 7: Replace the worst individuals with the best clones
        worst_indices = np.argsort(fitness)[-elite_size:]  # Get indices of the worst individuals
        population[worst_indices] = clones

        # Step 8: Output the best solution of the current generation
        best_solution = population[np.argmin(fitness)]  # Best individual has minimum fitness
        print(f"Generation {gen+1}, Best Solution: {best_solution}, Fitness: {rastrigin(best_solution)}")

# Parameters for the algorithm
population_size = 20  # Number of individuals
generations = 50  # Number of generations to evolve
mutation_rate = 0.1  # Chance of mutation in each clone
elite_size = 4  # Number of elite individuals to keep

# Run Clonal Selection Algorithm
clonal_selection_algorithm(population_size, generations, mutation_rate, elite_size)
