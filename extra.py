import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

MAX_GENERATIONS = 500
NUM_USERS = 943
NUM_MOVIES = 1682

rng = np.random.default_rng()

ratings_base = np.genfromtxt('ml-100k/ua.base', dtype='int32')
ratings_test = np.genfromtxt('ml-100k/ua.test', dtype='int32')
matrix_base = np.zeros((NUM_USERS, NUM_MOVIES), dtype='int32')
matrix_test = np.zeros((NUM_USERS, NUM_MOVIES), dtype='int32')

for r in ratings_base:
    matrix_base[r[0]-1, r[1]-1] = r[2]

for r in ratings_test:
    matrix_test[r[0] - 1, r[1] - 1] = r[2]

matrix_filled = matrix_base.copy()
unrated = np.where(matrix_filled == 0)
matrix_filled[unrated] = np.random.randint(low=1, high=6, size=len(unrated[1]))
rated_pos = np.where(matrix_base != 0)

user_ratings = []
neighborhoods = []
neighborhoods_ratings = []

for user in range(NUM_USERS):
    ratings = np.where(matrix_base[user] != 0)
    # Pair wise similarity with every user
    similarity = np.array([pearsonr(matrix_filled[user][ratings], us[ratings])[0] for us in matrix_filled])
    user_rated = matrix_filled[user][ratings]
    similarity = np.empty(NUM_USERS)
    for i, us in enumerate(matrix_filled):
        us_rated = us[ratings]
        pr = pearsonr(user_rated, us_rated)[0]
        if math.isnan(pr):
            pr = 0
        similarity[i] = pr

    max_users = similarity.argsort()[-11:-1][::-1]  # Top 10 most similar users most similar one is user itself

    neighborhood = matrix_base[max_users]  # Neighborhood of user
    # Actually rated movies of neighborhood
    neighborhood_rated = [np.where(neigh != 0) for neigh in neighborhood]
    neighborhoods.append(neighborhood)
    neighborhoods_ratings.append(neighborhood_rated)


def fitness(x):
    sum_pr = 0
    for usr in range(NUM_USERS):
        for i, neigh in enumerate(neighborhoods[usr]):
            x_red = x[usr][neighborhoods_ratings[usr][i]]
            neigh_red = neigh[neighborhoods_ratings[usr][i]]
            pr = pearsonr(x_red, neigh_red)[0]
            if math.isnan(pr):
                pr = 0
            sum_pr += pr
    return sum_pr
    # return sum([sum([
    #     pearsonr(x[usr][neighborhoods_ratings[usr][i]], neigh[neighborhoods_ratings[usr][i]])[0]
    #     for i, neigh in enumerate(neighborhoods[usr])]) for usr in range(NUM_USERS)])


def selection(pop_fitness):
    num_pairs = int(POPULATION_SIZE * PROBABILITY_OF_CROSSOVER / 2)    # Amount of parent pairs that should be selected
    parents = np.empty((num_pairs, 2), dtype='int32')
    for idx in range(num_pairs):
        sample = rng.choice(POPULATION_SIZE, TOURNAMENT_SIZE, replace=False)
        max_pair = pop_fitness[sample].argsort()[-2:][::-1]
        parents[idx] = sample[max_pair]
    return parents


def crossover(parent_pairs):
    offspring = []
    for parent in parent_pairs:
        positions = np.random.randint(low=0, high=2, size=(NUM_USERS, NUM_MOVIES))
        offspring.append(np.where(positions == 0, parent[0], parent[1]))
        offspring.append(np.where(positions == 0, parent[1], parent[0]))
    return offspring


def mutation(chromosomes):
    num_mutations = int(PROBABILITY_OF_MUTATION * NUM_MOVIES * NUM_USERS)
    for chrom in chromosomes:
        positions = rng.choice(NUM_MOVIES * NUM_USERS, num_mutations, replace=False)
        cols = (positions // NUM_USERS).reshape((1, num_mutations))
        rows = (positions % NUM_USERS).reshape((1, num_mutations))
        positions = np.concatenate((rows, cols), axis=0)
        values = np.random.randint(1, 6, num_mutations)
        chrom[tuple(positions)] = values
        chrom[rated_pos] = matrix_base[rated_pos]


combs = [(0.6, 0.00, 20, 5), (0.6, 0.001, 20, 5), (0.6, 0.1, 20, 5), (0.9, 0.01, 20, 5), (0.1, 0.01, 20, 5),
         (0.6, 0.00, 200, 20), (0.6, 0.01, 200, 20), (0.1, 0.01, 200, 20), (0.9, 0.01, 200, 20)]
print('RUN | Generations | Fitness | RMSE | MAE')
for ind, comb in enumerate(combs):
    PROBABILITY_OF_CROSSOVER, PROBABILITY_OF_MUTATION, POPULATION_SIZE, TOURNAMENT_SIZE = comb

    best = np.empty((MAX_GENERATIONS, 10))
    rmse = np.empty((MAX_GENERATIONS, 10))
    mae = np.empty((MAX_GENERATIONS, 10))
    optimals = []
    generations = []
    for i in range(10):
        # Create initial population
        initial_population = np.random.randint(low=1, high=6, size=(POPULATION_SIZE, NUM_USERS, NUM_MOVIES))
        for chromosome in initial_population:
            chromosome[rated_pos] = matrix_base[rated_pos]       # Inserts users actual ratings

        next_pop = initial_population
        next_pop_fitness = np.array([fitness(x) for x in next_pop])
        next_sorted_keys = next_pop_fitness.argsort()
        generation = 0
        last_leader_change = 0
        earlier_performance = 0
        finished = False
        while not finished:
            current_pop = next_pop
            current_pop_fitness = next_pop_fitness
            current_sorted_keys = next_sorted_keys
            parents = selection(current_pop_fitness)
            children = crossover(current_pop[parents])
            mutation(children)

            next_pop = current_pop.copy()
            next_pop[current_sorted_keys[:len(children)]] = np.array(children)  # Replace worst solutions with new ones
            children_fitness = np.array([fitness(x) for x in children])

            next_pop_fitness = np.empty_like(current_pop_fitness)
            next_pop_fitness[current_sorted_keys[len(children):]] = current_pop_fitness[current_sorted_keys[len(children):]]
            next_pop_fitness[current_sorted_keys[:len(children)]] = children_fitness

            next_sorted_keys = next_pop_fitness.argsort()
            improvement = np.sum(next_pop_fitness) / np.sum(current_pop_fitness)
            best_sol = next_pop[next_sorted_keys[-1]]
            best_fitness = next_pop_fitness[next_sorted_keys[-1]]
            best[generation][i] = best_fitness / NUM_USERS
            rmse[generation][i] = np.sqrt(mean_squared_error(best_sol[rated_pos], matrix_test[rated_pos]))
            mae[generation][i] = mean_absolute_error(best_sol[rated_pos], matrix_test[rated_pos])

            finished = generation > MAX_GENERATIONS or generation - last_leader_change > 50
            if not finished and generation >= 100:
                finished = best[generation, i] / best[generation - 100, i] < 1.001
            generation += 1

        best[generation:, i] = max(next_pop_fitness)
        rmse[generation:, i] = rmse[generation-1, i]
        mae[generation:, i] = mae[generation-1, i]
        optimals.append(max(next_pop_fitness))
        generations.append(generation)

    best_avg = np.average(best[:max(generations)], axis=1)
    rmse_avg = np.average(rmse[:max(generations)], axis=1)
    mae_avg = np.average(mae[:max(generations)], axis=1)
    # print('RUN', ind)
    # print('Average optimal', np.average(np.array(optimals)))
    # print('Average generations', np.average(np.array(generations)))
    # print('Average RMSE', rmse_avg[-1])
    # print('Average MAE', mae_avg[-1])
    print(ind, '&', np.average(np.array(generations)), '&', np.average(np.array(optimals)), '&', rmse_avg[-1], '&', mae_avg[-1], '\\\\')
    # print('----------------------------')

    plt.plot(np.arange(max(generations)), best_avg)
    plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
              + str(PROBABILITY_OF_MUTATION) + '%')
    plt.xlabel('Number of generations')
    plt.ylabel('Average fitness of best value')
    plt.grid(b=True, axis='y')
    plt.show()

    plt.plot(np.arange(max(generations)), rmse_avg)
    plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
              + str(PROBABILITY_OF_MUTATION) + '%')
    plt.xlabel('RMSE')
    plt.ylabel('Average RMSE of best value')
    plt.grid(b=True, axis='y')
    plt.show()

    plt.plot(np.arange(max(generations)), mae_avg)
    plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
              + str(PROBABILITY_OF_MUTATION) + '%')
    plt.xlabel('MAE')
    plt.ylabel('Average MAE of best value')
    plt.grid(b=True, axis='y')
    plt.show()

