import numpy as np
import random


class GeneticAlgorithm:
    """
    Genetic algorithm class for non derivative based optimsation over
    continuous parameters.
    """

    def __init__(self, param_constraints, pop_size=10, parent_size=5, mutate_rate=0.001):
        """
        :param param_constraints: N x 2 numpy array of parameter limits
        :param pop_size: Size of population at each generation
        """
        self.param_constraints = param_constraints
        self.pop_size = pop_size
        self.parent_size = parent_size
        self.mutate_rate = mutate_rate

        # set up the initial generation
        self.current_generation = np.random.uniform(0, 1,
            (self.pop_size, self.param_constraints.shape[0]))
        self.current_generation *= (self.param_constraints[:, 1]
                                    - self.param_constraints[:, 0])
        self.current_generation += self.param_constraints[:, 0]

    def new_generation(self, generation_scores=None):
        """
        Method to get the new generation of individuals
        :param generation_scores: Score for individuals from last generation
        :return: new_Generation: New generation of individuals to be tested
        """

        # If no cost function given then return current generation
        if generation_scores is None:
            return self.current_generation

        maters = np.zeros(self.parent_size, dtype=int)
        sum_score = np.sum(generation_scores)

        # Select maters based on cost function value. This is daintily not
        # an efficient way
        index = 0
        mater_index = 0
        while mater_index < self.parent_size:
            ran = random.uniform(0, sum_score)
            if generation_scores[index] > ran:
                maters[mater_index] = index
                mater_index += 1
            index = (index + 1) % self.pop_size

        # Generate new generation
        new_generation = np.empty_like(self.current_generation)
        for i in range(self.pop_size):
            if random.uniform(0, 1) < self.mutate_rate:
                # Random mutation
                child = ((self.param_constraints[:, 1]
                         - self.param_constraints[:, 0]) * random.uniform(0, 1)
                         + self.param_constraints[:, 0])
            else:
                # mate two parents
                mater1 = random.randint(0, self.parent_size)
                mater2 = random.randint(0, self.parent_size)
                child = self._mate(self.current_generation[mater1],
                                   self.current_generation[mater2])
            new_generation[i, :] = child

        self.current_generation = np.copy(new_generation)
        return self.current_generation

    @staticmethod
    def _mate(parent1, parent2):
        """
        Average parents params. Extend this
        """
        return (parent1 + parent2) / 2.0


if __name__ == '__main__':
    param_lims = np.array([-4., 4.])
    GA = GeneticAlgorithm(param_lims[None, :])

    gen = GA.new_generation()
    cost = -1.0 * gen**2.0 + 10.
    for x in range(3):
        gen = GA.new_generation(cost)
        cost = -1.0 * gen ** 2.0 + 10

    print(gen)