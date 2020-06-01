import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


PROBABILITY_OF_CROSSOVER = 0.9
PROBABILITY_OF_MUTATION = 0.01
POPULATION_SIZE = 200
TOURNAMENT_SIZE = 20
MAX_GENERATIONS = 1000
NUM_USERS = 943
NUM_MOVIES = 1682

rng = np.random.default_rng()

ratings = np.genfromtxt('ml-100k/u.data', dtype='int32')
matrix = np.zeros((NUM_USERS, NUM_MOVIES), dtype='int32')

for r in ratings:
    matrix[r[0]-1, r[1]-1] = r[2]

matrix_filled = matrix.copy()
unrated = np.where(matrix_filled == 0)
matrix_filled[unrated] = np.random.randint(low=1, high=6, size=len(unrated[1]))

user = matrix[0]                                    # User we are evaluating
user_rated = np.where(user != 0)
user_unrated = np.where(user == 0)
others = matrix[1:]
others_filled = matrix_filled[1:]                   # Other users with randomly filled values

# Pair wise similarity with every user
similarity = np.array([pearsonr(user[user_rated], us[user_rated])[0] for us in others_filled])

max_users = similarity.argsort()[-10:][::-1]        # Top 10 most similar users

neighborhood = others[max_users]                    # Neighborhood of user
# Actually rated movies of neighborhood
neighborhood_rated = [np.where(neigh != 0) for neigh in neighborhood]


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
        zeros = np.where(positions == 0)
        ones = np.where(positions == 1)
        off_1 = np.empty(NUM_MOVIES, dtype='int32')
        off_2 = np.empty(NUM_MOVIES, dtype='int32')
        off_1[zeros] = parent[0][zeros]
        off_1[ones] = parent[1][ones]
        off_2[zeros] = parent[1][zeros]
        off_2[ones] = parent[0][ones]
        offspring.extend([off_1, off_2])
    return offspring


def mutation(chromosomes):
    num_mutations = int(PROBABILITY_OF_MUTATION * NUM_MOVIES)
    for chrom in chromosomes:
        positions = rng.choice(NUM_MOVIES, num_mutations, replace=False)
        values = np.random.randint(1, 6, num_mutations)
        chrom[positions] = values
        chrom[user_rated] = user[user_rated]


best = np.empty((MAX_GENERATIONS, 10))
optimals = []
generations = []

for i in range(10):
    # Create initial population
    initial_population = np.random.randint(low=1, high=6, size=(POPULATION_SIZE, NUM_MOVIES))
    for chromosome in initial_population:
        chromosome[user_rated] = user[user_rated]       # Inserts users actual ratings

    next_pop = initial_population
    next_pop_fitness = np.array([fitness(p) for p in next_pop])
    generation = 0
    last_leader_change = 0
    improvement = 1
    while generation < MAX_GENERATIONS and generation - last_leader_change < 50:
        current_pop = next_pop
        current_pop_fitness = next_pop_fitness
        parents = selection(current_pop_fitness)
        children = crossover(current_pop[parents])
        mutation(children)

        next_pop = current_pop.copy()
        next_pop[current_pop_fitness.argsort()[:len(children)]] = np.array(children)
        next_pop_fitness = np.array([fitness(p) for p in next_pop])
        improvement = np.sum(next_pop_fitness) / np.sum(current_pop_fitness)
        last_leader_change = generation if max(current_pop_fitness) < max(next_pop_fitness) else last_leader_change
        print('Run:', i, 'Generation:', generation, 'Fitness', max(next_pop_fitness), 'Improvement:', improvement)

        best[generation][i] = max(next_pop_fitness)

        generation += 1

    best[generation:, i] = max(next_pop_fitness)
    optimals.append(max(next_pop_fitness))
    generations.append(generation)

print('Average optimal', np.average(np.array(optimals)))
print('Average generations', np.average(np.array(generations)))

best_avg = np.average(best[:max(generations)], axis=1)
plt.plot(np.arange(max(generations)), best_avg)
plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
          + str(PROBABILITY_OF_MUTATION) + '%')
plt.xlabel('Number of generations')
plt.ylabel('Average fitness of best value')
plt.grid(b=True, axis='y')
plt.show()
