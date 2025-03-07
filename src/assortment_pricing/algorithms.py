import abc
import numpy as np
import cvxpy as cp

from assortment_pricing import utils


class Algorithm(abc.ABC):
    @abc.abstractmethod
    def get_assortment_and_pricing(self, *args, **kwargs):
        pass

    def selection_feedback(self, *args, **kwargs):
        return

class OfflineOptimalAlgorithm(Algorithm):
    def __init__(self, K):
        self.K = K

    def get_assortment_and_pricing(self, alpha_star, beta_star):
        return utils.solve_assortment_and_pricing(self.K, alpha_star, beta_star)

class LinearPricingAlgorithm(Algorithm):
    def __init__(self, n, d, K, L0, T0, pool):
        self.pool = pool
        self.K = K
        self.n = n
        self.d = d
        self.theta = 0.5 * np.ones(2 * d)
        self.alpha_g = 0.01
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
            psi, phi = self.theta[:self.d], self.theta[self.d:]
            alpha = contexts[:, :self.d] @ psi
            beta = contexts[:, self.d:] @ phi
            V_inv = np.linalg.inv(self.V)
            linear_bonus = np.sqrt(self.alpha_g) * np.einsum('ij,jk,ki->i', contexts[:, :self.d], V_inv[:self.d, :self.d], contexts[:, :self.d].T)
            assortment, prices = utils.solve_assortment_and_pricing(self.K, alpha + linear_bonus, beta)
        return assortment, prices
    

class NonlinearPricingAlgorithm(Algorithm):
    def __init__(self, n, d, K, L0, T0, pool):
        self.pool = pool
        self.K = K
        self.n = n
        self.d = d
        self.theta = 0.5 * np.ones(2 * d)
        self.alpha_g = 0.01
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
            psi, phi = self.theta[:self.d], self.theta[self.d:]
            # define parameters of h(p ; h_p) = h_p[0] - h_p[1] p + sqrt{ h_p[2] - 2 * h_p[3] p + h_p[4] p^2 }.
            h_p = np.zeros((self.n, 5))
            V_inv = np.linalg.inv(self.V)
            h_p[:, 0] = contexts[:, :self.d] @ psi
            h_p[:, 1] = contexts[:, self.d:] @ phi
            h_p[:, 2] = np.sqrt(self.alpha_g) * np.einsum('ij,jk,ki->i', contexts[:, :self.d], V_inv[:self.d, :self.d], contexts[:, :self.d].T)
            h_p[:, 3] = np.sqrt(self.alpha_g) * np.einsum('ij,jk,ki->i', contexts[:, :self.d], V_inv[:self.d, self.d:], contexts[:, self.d:].T)
            h_p[:, 4] = np.sqrt(self.alpha_g) * np.einsum('ij,jk,ki->i', contexts[:, self.d:], V_inv[self.d:, self.d:], contexts[:, self.d:].T)
            assortment, prices = utils.solve_assortment_and_pricing_nonlinear(self.K, h_p, self.L0, self.pool)
        return assortment, prices

class DynamicAssortmentPricing(NonlinearPricingAlgorithm):
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


class NewtonAssortmentPricing(NonlinearPricingAlgorithm):

    def __init__(self, n, d, K, L0, T0, pool):
        self.theta0 = np.zeros(2 * d)
        self.alpha_g = 0.001
        super().__init__(n, d, K, L0, T0, pool)

    def mle_online_update(self, offered_contexts, selected_contexts):
        utilities = offered_contexts @ self.theta
        terms = np.exp(utilities)
        probs = terms / (1 + np.sum(terms))
        g = np.sum((probs * offered_contexts.T).T, axis=0) - selected_contexts
        V_half = np.linalg.cholesky(self.V)

        theta_param = cp.Variable(2 * self.d)
        obj = cp.Minimize(cp.sum_squares(theta_param @ V_half) + theta_param @ (8 * g - 2 * self.V @ self.theta))
        constraints = [cp.sum_squares(theta_param - self.theta0) <= min(np.log(2) * self.L0 / (8 * np.log(self.K)), 1) / 2]
        prob = cp.Problem(obj, constraints)
        prob.solve()

        theta = np.array(theta_param.value)

        return theta

    def selection_feedback(self, i_t, contexts, assortment, prices):
        x_tilde = contexts[assortment]
        x_tilde[:, self.d:] = - x_tilde[:, self.d:] * prices[assortment, np.newaxis]
        if self.t < self.T0:
            self.offered_contexts.append(x_tilde)
            if i_t is not None:
                self.selected_contexts.append(np.concatenate([contexts[i_t, :self.d], - prices[i_t] * contexts[i_t, self.d:]]))
            else:
                self.selected_contexts.append(np.zeros(2 * self.d))
            self.V += (1 / self.K ** 2) * x_tilde.T @ x_tilde
        if self.t == self.T0:
            self.theta = utils.solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta, scaling=self.L0 ** 2 )
            self.theta0 = self.theta.copy()
        if self.t >= self.T0:
            offered_contexts = x_tilde
            if i_t is not None:
                selected_contexts = np.concatenate([contexts[i_t, :self.d], - prices[i_t] * contexts[i_t, self.d:]])
            else:
                selected_contexts = np.zeros(2 * self.d)
            self.theta = self.mle_online_update(offered_contexts, selected_contexts)
            utilities = x_tilde @ self.theta
            probs = np.exp(utilities) / (1 + np.sum(np.exp(utilities)))
            Sigma = np.diag(probs) - np.outer(probs, probs)
            self.V += x_tilde.T @ Sigma @ x_tilde
        self.t += 1
