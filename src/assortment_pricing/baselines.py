import numpy as np
import cvxpy as cp

from assortment_pricing import utils, algorithms

class JavanmardDynamicPricing(algorithms.Algorithm):
    """
    Implementation of M3P (Javanmard et al., 2020)
    """
    def __init__(self, n, d, K, L0):
        self.K = K
        self.n = n
        self.d = d
        self.theta = np.zeros(2 * d)
        self.alpha_g = 0.003
        self.offered_contexts = []
        self.selected_contexts = []
        self.t = 0
        self.L0 = L0
        self.episode_len = d
        self.episode_t = 0
        self.V = np.zeros((2 * d, 2 * d))

    def get_assortment_and_pricing(self, contexts):
        if self.episode_t < self.d:
            assortment = np.random.choice(self.n, size=self.K, replace=False)
            prices = np.random.choice(2, size=self.n)
        else:
            psi, phi = self.theta[:self.d], self.theta[self.d:]
            alpha = np.minimum(contexts[:, :self.d] @ psi, 1)
            beta = np.maximum(contexts[:, self.d:] @ phi, self.L0)
            # compute prices assuming all items can be offered
            _, prices = utils.solve_assortment_and_pricing(self.n, alpha, beta)
            # choose best up-to-K items that result in the largest expected revenue
            values = alpha - beta @ prices
            assortment = []
            best_exp_rev = 0
            for K_counter in range(1, self.K+1):
                assortment_k = np.argpartition(values, -K_counter)[-K_counter:]
                expected_revenue_k = np.sum(prices[assortment_k] * np.exp(values[assortment_k])) \
                                   / (1 + np.sum(np.exp(values[assortment_k])))
                if expected_revenue_k > best_exp_rev:
                    assortment = assortment_k
                    best_exp_rev = expected_revenue_k
            # logging.debug(assortment)
            # logging.debug(prices)
        return assortment, prices

    def selection_feedback(self, i_t, contexts, assortment, prices):
        if self.episode_t < self.d:
            x_tilde = contexts[assortment]
            x_tilde[:, self.d:] = - x_tilde[:, self.d:] * prices[assortment, np.newaxis]
            self.V += x_tilde.T @ x_tilde
            self.offered_contexts.append(x_tilde)
            if i_t is not None:
                self.selected_contexts.append(np.concatenate([contexts[i_t, :self.d], - prices[i_t] * contexts[i_t, self.d:]]))
            else:
                self.selected_contexts.append(np.zeros(2 * self.d))
        if self.episode_t == self.d - 1:
            self.theta = utils.solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta, scaling=self.L0 ** 2)
        self.t += 1
        self.episode_t += 1
        # logging.debug(len(self.selected_contexts))
        if self.episode_t == self.episode_len:
            self.t = 0
            self.episode_t = 0
            self.episode_len += 1


class OhIyengarAssortmentSelection(algorithms.Algorithm):
    """
    Implementation of DBL-MNL (Oh & Iyengar, 2021)
    """
    def __init__(self, n, d, K, L0, T0, fixed_prices):
        self.K = K
        self.n = n
        self.d = d
        self.theta = np.zeros(d)
        self.alpha_g = 0.003
        self.offered_contexts = []
        self.selected_contexts = []
        self.t = 0
        self.L0 = L0
        self.T0 = T0
        self.V = np.zeros((d, d))
        self.fixed_prices = fixed_prices

    def get_assortment_and_pricing(self, contexts):
        if self.t < self.T0:
            assortment = np.random.choice(self.n, size=self.K, replace=False)
        else:
            g = np.zeros(self.n)
            V_inv = np.linalg.inv(self.V)
            for i in range(self.n):
                g[i] = self.alpha_g * np.inner(V_inv @ contexts[i, :self.d], contexts[i, :self.d])
            values = np.minimum(contexts[:, :self.d] @ self.theta + g, 1 + self.fixed_prices)
            assortment = []
            best_prob_selection = 0
            for K_counter in range(1, self.K + 1):
                assortment_k = np.argpartition(values, -K_counter)[-K_counter:]
                probability_of_selection_k = 1 - 1 / (1 + np.sum(np.exp(values[assortment_k])))
                if probability_of_selection_k > best_prob_selection:
                    assortment = assortment_k
                    best_prob_selection = probability_of_selection_k
        return assortment, self.fixed_prices * np.ones(self.n)

    def selection_feedback(self, i_t, contexts, assortment, prices):
        x_tilde = contexts[assortment, :self.d]
        self.V += x_tilde.T @ x_tilde
        self.offered_contexts.append(x_tilde)
        if i_t is not None:
            self.selected_contexts.append(contexts[i_t, :self.d])
        else:
            self.selected_contexts.append(np.zeros(self.d))
        if self.t >= self.T0:
            self.theta = utils.solve_mle(self.offered_contexts, self.selected_contexts, self.d, init_theta=self.theta)
        self.t += 1


class OhIyengarWithPricing(algorithms.LinearPricingAlgorithm):
    """
    Implementation of DBL-MNL (Oh & Iyengar, 2021) + Heuristic Dynamic Pricing
    """
    def __init__(self, n, d, K, L0, T0, pool):
        super().__init__(n, d, K, L0, T0, pool)
        self.alpha_g = 0.1

    def selection_feedback(self, i_t, contexts, assortment, prices):
        x_tilde = contexts[assortment]
        x_tilde[:, self.d:] = - x_tilde[:, self.d:] * prices[assortment, np.newaxis]
        self.V += x_tilde.T @ x_tilde
        self.offered_contexts.append(x_tilde)
        if i_t is not None:
            self.selected_contexts.append(np.concatenate([contexts[i_t, :self.d], - prices[i_t] * contexts[i_t, self.d:]]))
        else:
            self.selected_contexts.append(np.zeros(2 * self.d))
        if self.t >= self.T0:
            self.theta = utils.solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta, scaling=self.L0 ** 2)
        self.t += 1

class GoyalPerivierDynamicPricing(algorithms.Algorithm):
    """
    Implementation of ONS-MPP (Perivier & Goyal, 2022)
    """
    def __init__(self, n, d, K, L0):
        self.K = K
        self.n = n
        self.d = d
        self.theta = np.zeros(2 * d)
        self.offered_contexts = []
        self.selected_contexts = []
        self.t = 0
        self.L0 = L0
        self.V = 0.1 * np.identity(2 * d)

    def _update(self, theta, V, offered_contexts, selected_contexts, lmb):
        utilities = offered_contexts @ theta
        terms = np.exp(utilities)
        probs = terms / (1 + np.sum(terms))
        G = np.sum((probs * offered_contexts.T).T, axis=0) - selected_contexts
        theta = theta - np.linalg.pinv(V + lmb * np.identity(len(V))) @ G
        return theta

    def _project(self, contexts):
        V_half = np.linalg.cholesky(self.V)
        theta_param = cp.Variable(2 * self.d)
        obj = cp.Minimize(cp.sum_squares(theta_param @ V_half))
        constraints = [contexts[:, self.d:] @ theta_param[self.d:] >= self.L0]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return np.array(theta_param.value)

    def get_assortment_and_pricing(self, contexts):
        # project the parameter
        self.theta = self._project(contexts)
        psi, phi = self.theta[:self.d], self.theta[self.d:]
        alpha = contexts[:, :self.d] @ psi
        beta = contexts[:, self.d:] @ phi
        # compute prices assuming all items can be offered
        _, prices = utils.solve_assortment_and_pricing(self.n, alpha, beta)
        # choose best up-to-K items that result in the largest expected revenue
        values = alpha - beta @ prices
        assortment = []
        best_exp_rev = 0
        for K_counter in range(1, self.K + 1):
            assortment_k = np.argpartition(values, -K_counter)[-K_counter:]
            expected_revenue_k = np.sum(prices[assortment_k] * np.exp(values[assortment_k])) \
                                 / (1 + np.sum(np.exp(values[assortment_k])))
            if expected_revenue_k > best_exp_rev:
                assortment = assortment_k
                best_exp_rev = expected_revenue_k
        random_shock = (2 * np.random.choice(2, len(prices)) - 1) / ((self.t + 1) ** (1/4))
        prices = prices + random_shock
        return assortment, prices

    def selection_feedback(self, i_t, contexts, assortment, prices):
        x_tilde = contexts[assortment]
        x_tilde[:, self.d:] = - x_tilde[:, self.d:] * prices[assortment, np.newaxis]
        utilities = x_tilde @ self.theta
        probs = np.exp(utilities) / (1 + np.sum(np.exp(utilities)))
        Sigma = np.diag(probs) - np.outer(probs, probs)
        self.V += x_tilde.T @ Sigma @ x_tilde
        offered_contexts = x_tilde
        if i_t is not None:
            selected_contexts = np.concatenate([contexts[i_t, :self.d], - prices[i_t] * contexts[i_t, self.d:]])
        else:
            selected_contexts = np.zeros(2 * self.d)
        # update the parameter
        self.theta = self._update(self.theta, self.V, offered_contexts, selected_contexts, 10 * self.d * np.log(self.t + 1))


class GaussianThompsonSampling(algorithms.Algorithm):
    """
    Implementation of the Thompson sampling based assortment selection algorithm from Oh & Iyengar (2019). 
    We use "Gaussian approximation" to estimate the posterior. We cannot perform "optimistic sampling" in the pricing setting.
    This algorithm is not designed for dynamic pricing and we use a simple heursitic to choose prices.
    """
    def __init__(self, n, d, K, L0, T0, pool):
        self.pool = pool
        self.K = K
        self.n = n
        self.d = d
        self.theta = 0.5 * np.ones(2 * d)
        self.alpha_c = 0.4
        self.offered_contexts = []
        self.selected_contexts = []
        self.t = 0
        self.L0 = L0
        self.T0 = T0
        self.V = 0 * np.identity(2 * d)

    def get_assortment_and_pricing(self, contexts):
        if self.t < self.T0:
            assortment = np.random.choice(self.n, size=self.K, replace=False)
            prices = np.random.uniform(6, 10, size=self.n)
        else:
            alpha_t = self.alpha_c * np.sqrt(self.d + 4 * np.log(self.t + 1))
            theta_sampled = np.random.multivariate_normal(self.theta, alpha_t**2 * np.linalg.inv(self.V))
            psi, phi = theta_sampled[:self.d], theta_sampled[self.d:]
            alpha = contexts[:, :self.d] @ psi
            beta = contexts[:, self.d:] @ phi
            assortment, prices = utils.solve_assortment_and_pricing(self.K, alpha, beta)
        return assortment, prices

    def selection_feedback(self, i_t, contexts, assortment, prices):
        x_tilde = contexts[assortment]
        x_tilde[:, self.d:] = - x_tilde[:, self.d:] * prices[assortment, np.newaxis]
        self.offered_contexts.append(x_tilde)
        if i_t is not None:
            self.selected_contexts.append(np.concatenate([contexts[i_t, :self.d], - prices[i_t] * contexts[i_t, self.d:]]))
        else:
            self.selected_contexts.append(np.zeros(2 * self.d))
        if self.t >= self.T0:
            utilities = x_tilde @ self.theta
            probs = np.exp(utilities) / (1 + np.sum(np.exp(utilities)))
            Sigma = np.diag(probs) - np.outer(probs, probs)
            self.V += x_tilde.T @ Sigma @ x_tilde
            self.theta = utils.solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta, scaling=self.L0 ** 2)
        else:
            self.V += (1 / self.K ** 2) * x_tilde.T @ x_tilde
        self.t += 1