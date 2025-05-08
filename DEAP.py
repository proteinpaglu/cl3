import random
import numpy as np
from deap import base, creator, tools, algorithms

# Fitness function: maximize x * sin(x)
def evaluate(individual):
    x = individual[0]
    return x * np.sin(x),  # Tuple needed

# Setup for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=1.0)
toolbox.register("select", tools.selTournament, tournsize=3)

# Main function
def main():
    pop = toolbox.population(n=10)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

    best = tools.selBest(pop, k=1)[0]
    print("\nBest x:", best[0])
    print("Best fitness:", evaluate(best)[0])

main()
