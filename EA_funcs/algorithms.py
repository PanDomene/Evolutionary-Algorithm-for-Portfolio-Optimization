import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

class EA:
    """Evolutionary Algorithm for investment portfolio optimization."""

    def __init__(self, data, pop_size=100, lambda_=600, p_m=0.01, sigma=0.1,
                 max_w=0.1, delta=0.5, alpha=0.5, testing=True, max_risk=None,
                validation_split=0.3):
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
        - testing (bool, optional): If true, keeps track of several quantities
          for plotting, evaluating and debugging. (default True)
        """
        split = int((1 - validation_split) * len(data))
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.val_returns_pct = self.val_data.pct_change().dropna().values
        self.stats_(self.train_data) # Expected anual returns returns and covariance matrix
        self.pop_size = pop_size # Population size
        self.lambda_ = lambda_ # Number of offspring per generation
        self.p_m = p_m # Mutation rate
        self.sigma = sigma # Mutation step size
        self.delta = delta # Fitness weight constant
        self.alpha=0.5
        self.max_w = max_w # Max weight per asset
        self.assets = data.columns.values # Asset names
        self.n_assets = data.shape[1] # Number of assets
        self.max_risk = max_risk
        self.population = self.initialize_population_()
        self.fitness_evaluations = 0
        self.pop_fitness = self.get_fitness(self.population)
        self.best_chrom = self.population[np.argmax(self.pop_fitness)][:-1]
        
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
        - data (pd.DataFrame): Historical closing prices.
        """
        daily_returns = data.pct_change().dropna().values # daily % returns
        self.expected_returns = np.mean(daily_returns, axis=0) * 252 # Expected annual returns 
        self.cov_matrix = np.cov(daily_returns.T) # Covariance matrix of daily returns

    def volatility(self, weights):
        """
        Compute annualized volatility of a portfolio.
        
        Args:
            weights (np.ndarray): Vector of portfolio weights (no gene).
        
        Returns:
            float: Annualized volatility.
        """
        return np.sqrt(252 * weights @ self.cov_matrix @ weights)

    def weight_penalty_(self, chromosome, w=40):
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
        - max_risk (float): The maximum risk allowed.
        Returns:
        - tuple: Fitness value (f), expected returns for the portfolio (f1),
        and portfolio volatility (f2).
        """

        # If there is a tournament size gene
        if chromosome.shape[0] != self.cov_matrix.shape[0]:
            chromosome = chromosome[:-1] # ignore it
        
        self.fitness_evaluations += 1 # For performance evaluation (AES, SR)

        risk = self.volatility(chromosome) # The risk of the portfolio.

        # Return from validation data
        val_returns = self.val_returns_pct @ chromosome
        return_ = (1 + val_returns).prod() - 1 # Validation portfolio returns.
        fit = (1 - self.delta)*return_ - self.delta * risk # Fitness.
        
        if np.any(chromosome > self.max_w): # weight penalization
            fit -= self.weight_penalty_(chromosome)

        if self.max_risk and risk > self.max_risk:
            return -np.inf, 0, 0  # Invalid chromosome
            
        return fit, return_, risk
    


    def batch_fitness(self, group):
        """
        Vectorized fitness calculation for a group of chromosomes.
        
        Args:
            group (np.ndarray): Shape (n_chromosomes, n_assets + 1).
        
        Returns:
            np.ndarray: Fitness values for each chromosome.
        """

        self.fitness_evaluations += group.shape[0]

        chromosomes = group[:, :-1]  # remove tournament gene
        n = chromosomes.shape[0]

        # Validation returns matrix (T, n_assets)
        R = self.val_returns_pct  # shape (T, A)
        
        # === Compute portfolio returns for all chromosomes ===
        # Resulting shape: (T, n_chromosomes)
        val_returns = R @ chromosomes.T

        # Portfolio total return (1 + r₁)(1 + r₂)... - 1, vectorized
        cumulative_returns = np.prod(1 + val_returns, axis=0) - 1  # shape: (n_chromosomes,)
        
        # === Compute volatility ===
        # portfolio variance: x.T Σ x for each x
        # Vectorized as diag(X @ Σ @ X.T)
        cov = self.cov_matrix
        risks = np.sqrt(252 * np.einsum('ij,jk,ik->i', chromosomes, cov, chromosomes))  # shape: (n_chromosomes,)

        # === Fitness calculation ===
        fitness = (1 - self.delta) * cumulative_returns - self.delta * risks

        # === Penalize overweight assets ===
        excess = np.clip(chromosomes - self.max_w, 0, None)
        penalties = 40 * excess.sum(axis=1)  # shape (n,)
        fitness -= penalties

        # === Hard constraint: max risk ===
        if self.max_risk is not None:
            invalid = risks > self.max_risk
            fitness[invalid] = -np.inf

        return fitness

    def get_fitness(self, group):
        """
        Calculate fitness values for a group of portfolios.

        Args:
        - group (np.ndarray): Group of portfolios.

        Returns:
        - np.ndarray: Fitness values.
        """
        return self.batch_fitness(group)

    def normalize(self, chromosome):
        """
        Normalize a chromosome so that its genes sum up to 1.

        Args:
        - chromosome (np.ndarray): Portfolio weights.

        Returns:
        - np.ndarray: Normalized portfolio weights.
        """
        chromosome[chromosome < 0] *= -1
        return chromosome/chromosome.sum()

    def initialize_population_(self):
        """
        Initialize the EA population with chromosomes that satisfy both the maximum
        weight per asset and the maximum portfolio risk constraints.
    
        Each chromosome is a portfolio: a vector of asset weights summing to 1,
        followed by a self-adaptive gene for tournament size. Chromosomes are only 
        accepted if all weights are ≤ `max_w` and total volatility ≤ `max_risk`.
    
        Returns:
            np.ndarray: An array of shape (pop_size, n_assets + 1), where each row
                        represents a valid portfolio (chromosome) plus the self-adaptive gene.
        """
        population = np.empty([self.pop_size, self.n_assets + 1])
    
        for k in range(self.pop_size):
            attempts = 0
            while True:
                # Generate random weights between 0 and max_w
                chromosome = np.random.uniform(0, self.max_w, self.n_assets)
                chromosome = self.normalize(chromosome)
    
                # Check constraints
                good_weights = np.all(chromosome <= self.max_w)
                risk = self.volatility(chromosome)
                valid_risk = self.max_risk is None or risk <= self.max_risk
    
                if good_weights and valid_risk:
                    break
    
                attempts += 1
                if attempts > 500:
                    print(f"[WARNING] Could not initialize valid chromosome #{k} after 500 attempts.")
                    break
                    
            # Append self-adaptive gene for tournament size
            ki = np.random.uniform(0, 1)
            chromosome = np.append(chromosome, ki)
    
            # Store valid or last attempted chromosome (even if invalid)
            population[k] = chromosome
    
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
            mutated_fit = self.get_fitness(mutated) # Get their fitness.

            ### Update population.
            k = max(2, int(np.ceil(np.sum(mutated[:, -1])))) # Tournament size.
            self.population, self.pop_fitness = self.survival_selection(mutated,
                                                                 mutated_fit, k)

            ### Update and store historical attributes.
            if self.testing:
                self.best_fitness = self.pop_fitness.max()
                self.diversity = len(np.unique(self.population[:,:-1], axis=0))
                self.diversity_history.append(self.diversity)
                self.max_fit_history.append(self.pop_fitness.max())
                self.mean_fit_history.append(self.pop_fitness.mean())

        best_idx = np.argmax(self.pop_fitness)  # fallback
        
        self.best_chrom = self.population[best_idx][:-1]
        
    def test_returns(self, test_data, chrom):
        """
        Evaluate the portfolio's performance on test data using a given chromosome.
    
        Args:
            test_data (pd.DataFrame): Test set of asset closing prices.
            chrom (array-like): Portfolio weights (chromosome) representing the asset allocation.
    
        Returns:
            float: Final realized return on the test set in percent.
            pd.Series: Cumulative return time series.
        """

        returns = test_data.pct_change().dropna()
        port_returns = returns @ chrom
        cumulative = (1 + port_returns).cumprod() - 1
        actual_return = cumulative.iloc[-1]

        return actual_return, cumulative


    
    def plot_diversity(self, label=None, ax=None):
        """
        Plot the evolution of population diversity.
        """
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            show = True
        ax.plot(self.diversity_history, label=label)
        ax.set_title('Population Diversity')
        ax.set_ylabel('# of unique individuals')
        ax.set_xlabel('Generation')
        ax.set_yscale('log')
        if label:
            ax.legend()
        if show:
            plt.show()

    def plot_fitness(self, ax=None):
        """
        Plot the evolution of mean and max fitness.
        """
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            show = True
        ax.plot(self.mean_fit_history, label='mean population fitness', lw=3)
        ax.plot(self.max_fit_history, '--', label='max population fitness', lw=2)
        ax.legend()
        ax.set_title('Mean and Max Fitness Evolution')
        ax.set_ylabel('Fitness')
        ax.set_xlabel('Generation')
        if show:
            plt.show()

    def plot_max_fitness(self, label=None):
        """
        Plot the evolution of max fitness.
        """
        plt.plot(self.max_fit_history, lw=2, label=label)
        plt.title('Max Fitness Evolution')
        plt.ylabel('Fitness')
        plt.xlabel('Generation')

    def portfolio(self):
        not_null = self.best_chrom > 0
        df = pd.DataFrame(self.best_chrom[not_null], self.assets[not_null]).T
        return df

def random_search(closing_prices, num_chromosomes=100_000, alpha=0.3, beta=0.06,
                       max_attempts=10_000):
    """
    Searches for the single best random portfolio under constraints on max weight and risk,
    by sampling random portfolios and selecting the one with highest expected return.

    Args:
        closing_prices (pd.DataFrame): Historical closing prices. 
        num_chromosomes (int): Number of portfolios to sample. Default is 100,000.
        alpha (float): Max weight allowed per asset. Default is 0.3.
        beta (float): Max allowed portfolio variance. Default is 0.06.
        max_attempts (int): Max tries per chromosome to find valid sample. Default is 10,000.

    Returns:
        np.ndarray: Best chromosome (1D array of asset weights).
    """
    returns = closing_prices.pct_change().dropna()
    cov_matrix = returns.cov().values
    expected_returns = returns.mean().values
    num_genes = len(expected_returns)

    best_chromosome = None
    best_return = -np.inf

    for _ in range(num_chromosomes):
        for _ in range(max_attempts):
            chromosome = np.random.rand(num_genes)
            chromosome /= chromosome.sum()

            variance = np.sqrt(252 * chromosome @ cov_matrix @ chromosome)

            if variance <= beta and np.all(chromosome <= alpha):
                ret = chromosome @ expected_returns
                if ret > best_return:
                    best_return = ret
                    best_chromosome = chromosome
                break
        else:
            raise RuntimeError("Could not generate a valid chromosome after max_attempts")

    return best_chromosome


    
    return population

# ---- Helper ----

def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


# ---- 1. Equal Weight ----
def equal_weight(closing_prices):
    n_assets = closing_prices.shape[1]
    return np.ones(n_assets) / n_assets


# ---- 2. Inverse Volatility ----
def inverse_volatility_weights(closing_prices):
    log_returns = compute_log_returns(closing_prices)
    vol = log_returns.std()
    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()
    return weights.reindex(closing_prices.columns, fill_value=0.0).values


# ---- 3. Minimum Variance Portfolio (long-only) ----
def minimum_variance_weights(closing_prices):
    log_returns = compute_log_returns(closing_prices)
    cov = log_returns.cov().values
    n = cov.shape[0]
    columns = closing_prices.columns

    def portfolio_variance(weights):
        return weights.T @ cov @ weights

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    initial_guess = np.ones(n) / n

    result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        raise ValueError("Minimum variance optimization failed")


# ---- 4. Hierarchical Risk Parity (HRP) ----
def correl_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

def get_quasi_diag(link):
    """
    Sort clustered items recursively to get the order of leaf indices.

    Args:
        link: Linkage matrix from scipy.cluster.hierarchy.linkage

    Returns:
        A list/array of leaf indices sorted according to hierarchical clustering.
    """
    def recursive_sort(link, node):
        if node < len(link) + 1:
            return [node]
        else:
            left = int(link[node - len(link) - 1, 0])
            right = int(link[node - len(link) - 1, 1])
            return recursive_sort(link, left) + recursive_sort(link, right)

    n = link.shape[0] + 1
    root_node = 2 * n - 2  # root node index in linkage
    return recursive_sort(link, root_node)

def hrp_weights(closing_prices):
    log_returns = compute_log_returns(closing_prices)
    corr = log_returns.corr()
    dist = correl_dist(corr)
    dist_mat = squareform(dist.values)
    link = linkage(dist_mat, method='single')
    sort_ix = get_quasi_diag(link)
    sorted_assets = log_returns.columns[sort_ix].tolist()

    cov = log_returns.cov()
    
    def recursive_bisection(cov, assets):
        if len(assets) == 1:
            return pd.Series(1.0, index=assets)
        else:
            split = len(assets) // 2
            left = assets[:split]
            right = assets[split:]
            w_left = recursive_bisection(cov, left)
            w_right = recursive_bisection(cov, right)
            var_left = np.dot(w_left.T, np.dot(cov.loc[left, left], w_left))
            var_right = np.dot(w_right.T, np.dot(cov.loc[right, right], w_right))
            alpha = 1.0 - var_left / (var_left + var_right)
            w_left *= alpha
            w_right *= 1.0 - alpha
            return pd.concat([w_left, w_right])

    weights_series = recursive_bisection(cov, sorted_assets)
    # Reindex to match original asset order
    weights_series = weights_series.reindex(closing_prices.columns, fill_value=0.0)
    return weights_series.values


############### Put it all together ################

def compute_all_weights(closing_prices, num_random=1000, alpha=0.3, beta=0.06, seed=None):

    return {
        "Equal Weight": equal_weight(closing_prices),
        "Inverse Volatility": inverse_volatility_weights(closing_prices),
        "Minimum Variance": minimum_variance_weights(closing_prices),
        "HRP": hrp_weights(closing_prices),
        "Random Portfolios": random_search(closing_prices, num_chromosomes=num_random, alpha=alpha, beta=beta)
    }