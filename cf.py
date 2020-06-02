import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


PROBABILITY_OF_CROSSOVER = 0.9
PROBABILITY_OF_MUTATION = 0.01
POPULATION_SIZE = 200
TOURNAMENT_SIZE = 20
MAX_GENERATIONS = 1000
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


def fitness(x):
    return sum([pearsonr(x[neighborhood_rated[i]], neigh[neighborhood_rated[i]])[0] for i, neigh in enumerate(neighborhood)])


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
        positions = np.random.randint(low=0, high=2, size=NUM_MOVIES)
        offspring.append(np.where(positions == 0, parent[0], parent[1]))
        offspring.append(np.where(positions == 0, parent[1], parent[0]))
    return offspring


def mutation(chromosomes):
    num_mutations = int(PROBABILITY_OF_MUTATION * NUM_MOVIES)
    for chrom in chromosomes:
        positions = rng.choice(NUM_MOVIES, num_mutations, replace=False)
        values = np.random.randint(1, 6, num_mutations)
        chrom[positions] = values
        chrom[user_rated] = user[user_rated]


random_users = rng.choice(NUM_USERS, 20, replace=False)
for ind, current_user in enumerate(random_users):
    user = matrix_base[current_user]  # User we are evaluating
    user_rated = np.where(user != 0)
    user_test_pos = np.where(matrix_test[current_user] != 0)
    user_test = matrix_test[current_user][user_test_pos]
    user_unrated = np.where(user == 0)
    others = np.delete(matrix_base, current_user, axis=0)
    others_filled = np.delete(matrix_filled, current_user, axis=0)  # Other users with randomly filled values

    # Pair wise similarity with every user
    similarity = np.array([pearsonr(user[user_rated], us[user_rated])[0] for us in others_filled])

    max_users = similarity.argsort()[-10:][::-1]  # Top 10 most similar users

    neighborhood = others[max_users]  # Neighborhood of user
    # Actually rated movies of neighborhood
    neighborhood_rated = [np.where(neigh != 0) for neigh in neighborhood]

    best = np.empty((MAX_GENERATIONS, 10))
    rmse = np.empty((MAX_GENERATIONS, 10))
    mae = np.empty((MAX_GENERATIONS, 10))
    optimals = []
    generations = []
    for i in range(10):
        # Create initial population
        initial_population = np.random.randint(low=1, high=6, size=(POPULATION_SIZE, NUM_MOVIES))
        for chromosome in initial_population:
            chromosome[user_rated] = user[user_rated]       # Inserts users actual ratings

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
            next_pop[current_sorted_keys[:len(children)]] = np.array(children)
            next_pop_fitness = np.array([fitness(x) for x in next_pop])
            next_sorted_keys = next_pop_fitness.argsort()
            improvement = np.sum(next_pop_fitness) / np.sum(current_pop_fitness)
            best_sol = next_pop[next_sorted_keys[-1]]
            best_fitness = next_pop_fitness[next_sorted_keys[-1]]
            last_leader_change = generation if current_pop_fitness[current_sorted_keys[-1]] < best_fitness else last_leader_change

            best[generation][i] = best_fitness
            rmse[generation][i] = np.sqrt(mean_squared_error(best_sol[user_test_pos], user_test))
            mae[generation][i] = mean_absolute_error(best_sol[user_test_pos], user_test)

            finished = generation > MAX_GENERATIONS or generation - last_leader_change > 50
            if not finished and generation >= 100:
                finished = best[generation, i] / best[generation - 100, i] < 1.01
            generation += 1

        best[generation:, i] = max(next_pop_fitness)
        rmse[generation:, i] = rmse[generation-1, i]
        mae[generation:, i] = mae[generation-1, i]
        optimals.append(max(next_pop_fitness))
        generations.append(generation)

    best_avg = np.average(best[:max(generations)], axis=1)
    rmse_avg = np.average(rmse[:max(generations)], axis=1)
    mae_avg = np.average(mae[:max(generations)], axis=1)
    print('RUN', ind, 'USER', current_user)
    print('Average optimal', np.average(np.array(optimals)))
    print('Average generations', np.average(np.array(generations)))
    print('Average RMSE', rmse_avg[-1])
    print('Average MAE', mae_avg[-1])
    print('----------------------------')

    # plt.plot(np.arange(max(generations)), best_avg)
    # plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
    #           + str(PROBABILITY_OF_MUTATION) + '%')
    # plt.xlabel('Number of generations')
    # plt.ylabel('Average fitness of best value')
    # plt.grid(b=True, axis='y')
    # plt.show()
    #
    # plt.plot(np.arange(max(generations)), rmse_avg)
    # plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
    #           + str(PROBABILITY_OF_MUTATION) + '%')
    # plt.xlabel('Number of generations')
    # plt.ylabel('Average RMSE of best value')
    # plt.grid(b=True, axis='y')
    # plt.show()
    #
    # plt.plot(np.arange(max(generations)), mae_avg)
    # plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
    #           + str(PROBABILITY_OF_MUTATION) + '%')
    # plt.xlabel('Number of generations')
    # plt.ylabel('Average MAE of best value')
    # plt.grid(b=True, axis='y')
    # plt.show()

