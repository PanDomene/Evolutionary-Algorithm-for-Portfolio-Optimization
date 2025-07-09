import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class EA:
    """Evolutionary Algorithm for investment portfolio optimization."""

    def __init__(self, data, pop_size=100, lambda_=600, p_m=0.01, sigma=0.1,
                 max_w=0.1, delta=0.5, alpha=0.5, testing=True):
        """
        Initialize the EA with the given parameters and data.

        Args:
        - data (pd.DataFrame): Historical closing prices.
        - pop_size (int, optional): Population size for the EA. (default 100)
        - lambda_ (int, optional): Number of offspring per generation.
          (default 600)
        - p_m (float, optional): Mutation rate. (default 0.01)
        - sigma (float, optional): Mutation step size. (default 0.1)
        - max_w (float, optional): Maximum weight per asset. (default 0.1)
        - delta (float, optional): Fitness weight constant. (default 0.5)
        - alpha (float, optional): Blend crossover constant. (default 0.5)
        """
        self.stats_(data) # Expected anual returns returns and covariance matrix
        self.pop_size = pop_size # Population size
        self.lambda_ = lambda_ # Number of offspring per generation
        self.p_m = p_m # Mutation rate
        self.sigma = sigma # Mutation step size
        self.delta = delta # Fitness weight constant
        self.alpha=0.5
        self.max_w = max_w # Max weight per asset
        self.assets = data.columns.values # Asset names
        self.n_assets = data.shape[1] # Number of assets
        self.population = self.initialize_population_()
        self.fitness_evaluations = 0
        self.pop_fitness = self.get_fitness(self.population[:,:-1])

        # For plotting and evaluating model performance and robustness.
        self.testing = testing
        if self.testing:
            self.diversity = len(np.unique(self.population, axis=0))
            self.diversity_history = [self.diversity]
            self.max_fit_history = [self.pop_fitness.mean()]
            self.mean_fit_history = [self.pop_fitness.max()]
            self.best_fitness = self.pop_fitness.max()

    def stats_(self, data):
        """
        Compute and store expected annual returns and covariance matrix.

        Args:
        - data (pd.DataFrame): Historical asset prices.
        """
        # Daily closing price % change.
        daily_change = data.pct_change()
        # Expected anual return.
        self.expected_returns = (daily_change.mean()*252).values
        # Covariance matrix.
        self.cov_matrix = daily_change.cov().to_numpy()

    def volatility(self, chromosome):
        """
        Calculate the volatility (risk) of the portfolio.

        Args:
        - chromosome (np.ndarray): Portfolio weights.

        Returns:
        - float: Portfolio volatility.
        """
        return np.sqrt(252*chromosome.T @ self.cov_matrix @ chromosome)

    def weight_penalty_(self, chromosome, w=20):
        """
        Compute penalization for weights exceeding the maximum allowed.

        Args:
        - chromosome (np.ndarray): Portfolio weights.
        - w (float, optional): Penalty weight.

        Returns:
        - float: Penalty value.
        """
        penalization = 0
        for gene in chromosome[:-1]:
            if gene > self.max_w: # If too high...
                penalization += gene - self.max_w # penalize proportionally.
        return w*penalization

    def fitness(self, chromosome):
        """
        Calculate the fitness of the portfolio.

        Args:
        - chromosome (np.ndarray): Portfolio weights.

        Returns:
        - tuple: Fitness value (f), expected returns for the portfolio (f1),
        and portfolio volatility (f2).
        """
        self.fitness_evaluations += 1 # For performance evaluation (AES, SR)
        f1 = chromosome @ self.expected_returns # Expected portfolio returns.
        f2 = self.volatility(chromosome) # The risk of the portfolio.
        f = (1 - self.delta)*f1 - self.delta * f2 # Fitness.
        if np.any(chromosome > self.max_w): # weight penalization
            f -= self.weight_penalty_(chromosome)
        return f, f1, 100*f2

    def get_fitness(self, group):
        """
        Calculate fitness values for a group of portfolios.

        Args:
        - group (np.ndarray): Group of portfolios.

        Returns:
        - np.ndarray: Fitness values.
        """
        return np.array([self.fitness(x)[0] for x in group])

    def normalize(self, chromosome):
        """
        Normalize a chromosome so that its genes sum up to 1.

        Args:
        - chromosome (np.ndarray): Portfolio weights.

        Returns:
        - np.ndarray: Normalized portfolio weights.
        """
        for i, gene in enumerate(chromosome):
            if gene < 0: # Portfolio cannot have negative weights.
                chromosome[i] = 0
        return chromosome/chromosome.sum()

    def initialize_population_(self):
        """
        Initialize the population satisfying problem constraints.

        Returns:
        - np.ndarray: Initial population.
        """
        population = np.empty([self.pop_size, self.n_assets + 1])
        for k in range(self.pop_size):
            while True: # Generate until valid portfolio is found.
                chromosome = np.random.uniform(0, self.max_w, self.n_assets)
                chromosome = self.normalize(chromosome)
                good_weights = np.all(chromosome <= self.max_w)
                if good_weights:
                    break
            ki = np.random.uniform(0,1) # Tournament size self-adaptive gene.
            chromosome = np.append(chromosome, ki) # Add self-adaptive gene.
            population[k] = chromosome # Add chromosome to population.
        return population

    def parent_selection(self):
        """
        Perform uniform parent selection.

        Returns:
        - np.ndarray: Selected parent population.
        """
        # Selection is performed WITH replacement.
        selected = np.random.randint(0, self.pop_size, self.lambda_)
        return self.population[selected]

    def recombination(self, p1, p2):
        """
        Perform blend crossover (BLX-alpha) recombination.

        Args:
        - p1 (np.ndarray): First parent chromosome.
        - p2 (np.ndarray): Second parent chromosome.

        Returns:
        - tuple: Two child chromosomes.
        """
        u = np.random.uniform()
        gamma = (1 - 2*self.alpha) * u - self.alpha
        child1 = (1 - gamma) * p1 + gamma * p2
        child2 = (1 - gamma) * p2 + gamma * p1
        return child1, child2

    def mutation(self, chromosome, eta=0.22):
        """
        Perform Gaussian perturbation mutation.

        Args:
        - chromosome (np.ndarray): Input chromosome.
        - eta (float, optional): Tournament size mutation parameter
          (default is 0.22).

        Returns:
        - np.ndarray: Mutated chromosome.
        """
        ### Mutation for portfolio weights.
        for i, gene in enumerate(chromosome[:-1]):
            if np.random.random() < self.p_m: # P(mutation) = p_m per gene.
                chromosome[i] += np.random.normal(scale=self.sigma)
        x = self.normalize(chromosome[:-1])

        # Mutation for (self-adaptive) tournament size gene.
        k = chromosome[-1]
        if k < 0 or k >= 1: # k has to be in [0, 1].
          k = np.random.uniform(0, 1)
        new_k = ( 1 + ( (1 - k) / k ) * np.exp(-eta*np.random.normal()) )**(-1)

        return np.append(x, new_k)

    def survival_selection(self, mutated, mutated_fitness, k):
        """
        Perform tournament replacement for survival selection.

        Args:
        - mutated (np.ndarray): Mutated population.
        - mutated_fitness (np.ndarray): Fitness values of mutated population.
        - k (int): Tournament size.

        Returns:
        - tuple: New generation and fitness values.
        """
        new_generation = np.empty([self.pop_size, self.n_assets+1])
        new_fitness = np.empty(self.pop_size)
        for i in range(self.pop_size):
            # Select k contestants.
            indices = np.random.choice(self.lambda_, k, replace=False)
            tourn = mutated[indices] # Their chromosomes.
            fit_tourn = mutated_fitness[indices] # And their fitness.
            win = tourn[np.argmax(fit_tourn)] # Highest fitness wins.
            win_fit = fit_tourn.max()
            new_generation[i] = win
            new_fitness[i] = win_fit
        return new_generation, new_fitness


    def run(self, iters):
        """
        Run the EA for a specified number of generations.

        Args:
        - iters (int): Number of generations to run.
        """
        for _ in range(iters):
            parents = self.parent_selection() # Select parents.
            offspring = np.empty([self.lambda_, self.n_assets+1])
            mutated = np.empty(offspring.shape)

            ### Produce offspring.
            for i in range(0, self.lambda_, 2):
                offspring[i], offspring[i+1] = self.recombination(parents[i],
                                                                   parents[i+1])
                mutated[i], mutated[i+1] = self.mutation(offspring[i]), \
                                           self.mutation(offspring[i+1])
            mutated_fit = self.get_fitness(mutated[:,:-1]) # Get their fitness.

            ### Update population.
            k = int(np.ceil(np.sum(mutated[:, -1]))) # Tournament size.
            self.population, self.pop_fitness = self.survival_selection(mutated,
                                                                 mutated_fit, k)

            ### Update and store historical attributes.
            if self.testing:
                self.best_fitness = self.pop_fitness.max()
                self.diversity = len(np.unique(self.population[:,:-1], axis=0))
                self.diversity_history.append(self.diversity)
                self.max_fit_history.append(self.pop_fitness.max())
                self.mean_fit_history.append(self.pop_fitness.mean())

    def plot_diversity(self, label=None):
        """
        Plot the evolution of population diversity.
        """
        plt.plot(self.diversity_history, label=label)
        plt.title('Population Diversity')
        plt.ylabel('# of unique individuals')
        plt.xlabel('Generation')

    def plot_fitness(self):
        """
        Plot the evolution of mean and max fitness.
        """
        plt.plot(self.mean_fit_history, label='mean population fitness', lw=3)
        plt.plot(self.max_fit_history, '--', label='max population fitness',
                                                                           lw=2)
        plt.legend()
        plt.title('Mean and Max Fitness Evolution')
        plt.ylabel('Fitness')
        plt.xlabel('Generation')

    def plot_max_fitness(self, label=None):
        """
        Plot the evolution of max fitness.
        """
        plt.plot(self.max_fit_history, lw=2, label=label)
        plt.title('Max Fitness Evolution')
        plt.ylabel('Fitness')
        plt.xlabel('Generation')

    def portfolio(self):
        best_portfolio = self.population[np.argmax(self.pop_fitness)][:-1]
        not_null = best_portfolio > 0
        df = pd.DataFrame(best_portfolio[not_null], self.assets[not_null]).T
        return df

def random_search(cov_matrix, num_chromosomes=100_000, alpha=0.3, beta=0.06):
    """
    Generates a population of random portfolios (chromosomes) that satisfy
    constraints on maximum risk and maximum weight per asset.

    Each portfolio is represented as a chromosome: a vector of asset weights 
    (genes) that sum to 1. Only chromosomes with all weights ≤ `alpha` and
    overall portfolio variance (i.e. risk) ≤ `beta` are accepted.

    Args:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        num_chromosomes (int): Number of valid portfolios to generate. Default is 100,000.
        alpha (float): Maximum allowed weight for any single asset. Default is 0.3.
        beta (float): Maximum allowed portfolio variance (risk). Default is 0.06.

    Returns:
        np.ndarray: Array of shape (num_chromosomes, num_assets), where each row is
        a valid portfolio (chromosome) satisfying the constraints.
    """
    num_genes = len(cov_matrix)
      
    population = np.zeros([num_chromosomes, num_genes])
    
    for i in range(num_chromosomes):
        while True:
            cromosome = np.random.rand(num_genes)
    
            # Normalise chromosome so that the sum of genes is 1
            cromosome /= np.sum(cromosome)
    
            # Find the porfolio's variance
            variance = np.dot(np.dot(cromosome, cov_matrix), cromosome.T)
    
            # Verify if chromosome satisfies max risk and max weight conditions
            if variance <= beta and  np.all(cromosome <= alpha):
                # if so, add that chromosome to the population
                population[i] = cromosome 
                break # and start generating a new one.

    return population
