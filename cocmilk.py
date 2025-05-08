import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import random

# Dummy Spray Drying Data (temperature, pressure, air flow -> moisture content)
data = np.random.rand(100, 3)
target = data[:, 0] * 0.5 + data[:, 1] * 0.3 + data[:, 2] * 0.2  # Simplified target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# Define GA Evaluation Function
def evaluate_nn(individual):
    # Ensure hidden layers have a minimum size of 1
    individual = [max(1, layer_size) for layer_size in individual]
    
    # Neural Network with individual as parameters
    nn = MLPRegressor(hidden_layer_sizes=(individual[0], individual[1]),
                      activation='relu', solver='adam', max_iter=2000)
    nn.fit(X_train, y_train)
    predictions = nn.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return (mse,)

# Set up Genetic Algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 100)  # Ensure hidden layer size is at least 1
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, toolbox.attr_int), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_nn)

# Custom mutation function to avoid 0 hidden layers
def mutate_valid(individual):
    # Apply mutation
    tools.mutFlipBit(individual, indpb=0.1)
    
    # Ensure hidden layer sizes are at least 1
    individual[0] = max(1, individual[0])
    individual[1] = max(1, individual[1])
    return individual,

toolbox.register("mutate", mutate_valid)

# Create Population and Run GA
population = toolbox.population(n=20)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=30, verbose=True)

# Output the best individual (optimized NN params)
best_individual = tools.selBest(population, 1)[0]
print("Best individual:", best_individual)
