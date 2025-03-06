import itertools
import logging
import time
import abc

import numpy as np
from functools import reduce
from scipy import optimize as opt
import cvxpy as cp


def solve_assortment_and_pricing(K, alpha, beta):
    B = 1
    assortment = np.array([])
    B_min = 0
    B_max = 1000
    while B_max - B_min > 1e-6:
        B = (B_max + B_min)/2
        v_of_B = np.exp(alpha - beta * B - 1) / beta
        assortment = np.argpartition(v_of_B, -K)[-K:]
        B_achieved = np.sum(v_of_B[assortment])
        if B_achieved < B:
            B_max = B
        else:
            B_min = B

    prices = 1 / beta + B
    return assortment, prices


def solve_assortment_and_pricing_nonlinear(K, h_p, L_0, pool):
    prices = np.zeros(len(h_p))
    assortment = np.array([])
    B_min = 0
    B_max = 7/L_0
    while B_max - B_min > 1e-6:
        B = (B_max + B_min)/2
        prices_and_v = pool.map(solve_v_value, itertools.product(h_p, [L_0], [B]))
        prices, v_of_B = zip(*prices_and_v)
        prices = np.array(prices)
        v_of_B = np.array(v_of_B)
        assortment = np.argpartition(v_of_B, -K)[-K:]
        B_achieved = np.sum(v_of_B[assortment])
        if B_achieved < B:
            B_max = B
        else:
            B_min = B
    return assortment, prices


def solve_v_value(inputs):

    h_p, L_0, B = inputs

    # print(h_p, "determinant = ", 4 * h_p[3] ** 2 - 4 * h_p[2] * h_p[4])

    epsilon = 1e-5
    # add small constant in the square root to avoid numerical instability
    h_p[2] = h_p[2] + 1e-3
    p_max = 14/L_0

    h = lambda p: h_p[0] - h_p[1] * p + np.sqrt(h_p[2] - 2 * h_p[3] * p + h_p[4] * p ** 2)

    h_1 = lambda p: (h_p[4] * p - h_p[3]) / np.sqrt(h_p[2] - 2 * h_p[3] * p + h_p[4] * p ** 2) - h_p[1]
    # h_2 = lambda p: (h_p[2] * h_p[4] - h_p[3] ** 2) / (h_p[2] - 2 * h_p[3] * p + h_p[4] * p ** 2)**(3/2)
    # h_3 = lambda p: 3 * (h_p[2] * h_p[4] - h_p[3] ** 2) * (h_p[3] - h_p[4] * p) / (h_p[2] - 2 * h_p[3] * p + h_p[4] * p ** 2)**(5/2)

    z = lambda p: 1 / (B - p)
    # z_1 = lambda p: 1 / (B - p) ** 2
    # z_2 = lambda p: 2 / (B - p) ** 3

    roots = []

    # find the point where h'(p) = -L_0

    if h_1(0) + L_0 > 0:
        roots.append(B + 1 / L_0)
        crit_p = B
    else:
        if h_1(p_max) + L_0 > 0:
            opt_result = opt.root_scalar(lambda p: h_1(p) + L_0, x0=h_p[3] / h_p[4], bracket=(0, p_max), method="brentq")
            crit_p = opt_result.root
            possible_root = B + 1 / L_0
            if possible_root >= crit_p:
                roots.append(possible_root)
        else:
            crit_p = p_max

        if B < crit_p:
            poly = np.zeros(7)
            poly[0] = h_p[4] ** 3 / (h_p[2] * h_p[4] - h_p[3] ** 2) ** 2
            poly[1] = - 6 * h_p[3] * h_p[4] ** 2 / (h_p[2] * h_p[4] - h_p[3] ** 2) ** 2
            poly[2] = (3 * h_p[2] * h_p[4] ** 2 + 12 * h_p[3] ** 2 * h_p[4]) / (h_p[2] * h_p[4] - h_p[3] ** 2) ** 2 - 1
            poly[3] = (- 12 * h_p[2] * h_p[3] * h_p[4] - 8 * h_p[3] ** 3) / (h_p[2] * h_p[4] - h_p[3] ** 2) ** 2 + 4 * B
            poly[4] = (3 * h_p[2] ** 2 * h_p[4] + 12 * h_p[2] * h_p[3] ** 2) / (h_p[2] * h_p[4] - h_p[3] ** 2) ** 2 - 6 * B ** 2
            poly[5] = (- 6 * h_p[2] ** 2 * h_p[3]) / (h_p[2] * h_p[4] - h_p[3] ** 2) ** 2 + 4 * B ** 3
            poly[6] = h_p[2] ** 3 / (h_p[2] * h_p[4] - h_p[3] ** 2) ** 2 - B ** 4

            crit_points = np.roots(poly)
            section_boundaries = [np.abs(x) for x in crit_points if np.isreal(x) and crit_p > x > B]
            section_boundaries = [B + epsilon] + sorted(section_boundaries) + [crit_p]

            fun = lambda p:  h_1(p) - z(p)

            for i in range(len(section_boundaries) - 1):
                p_1, p_2 = section_boundaries[i], section_boundaries[i + 1]
                if fun(p_1) * fun(p_2) < 0:
                    opt_result = opt.root_scalar(fun, x0=(p_1 + p_2)/2, bracket=(p_1, p_2), method="brentq")
                    roots.append(opt_result.root)

    roots = np.array(roots)

    def evaluate_f_function(p):
        if p < crit_p:
            return - np.exp(h(p)) / h_1(p)
        else:
            return - np.exp(h(crit_p) - L_0 * (p - crit_p)) / L_0

    if len(roots) > 0:
        f_values = np.array([evaluate_f_function(p) for p in roots])
        best_i = np.argmax(f_values)
        return roots[best_i], f_values[best_i]
    else:
        return 0, 0


def grad_reduce(input_grad, sum_grad):
    return sum_grad[0] + input_grad[0], sum_grad[1] + input_grad[1]


def solve_mle(offered_contexts, selected_contexts, d, init_theta=None, scaling=1):
    if init_theta is None or np.isnan(init_theta).any():
        theta = 0.5 * np.ones(d)
    else:
        theta = init_theta
    iter = 0
    while True:
        d_half = d // 2
        def grad_map(input_contexts):
            offered_contexts, selected_contexts = input_contexts
            utilities = offered_contexts @ theta
            terms = np.exp(utilities)
            probs = terms / (1 + np.sum(terms))
            grad_new = probs @ offered_contexts - selected_contexts
            W = np.diag(probs) - np.outer(probs, probs)
            hess_new = offered_contexts.T @ W @ offered_contexts
            return grad_new, hess_new

        results = map(grad_map, zip(offered_contexts, selected_contexts))
        grad, hess = reduce(grad_reduce, results)

        lmb = 5

        # reg_grad = lmb/d * theta
        # reg_grad[d_half:] = reg_grad[d_half:] / scaling
        # grad = grad + reg_grad

        reg_hess = lmb * np.identity(d)
        reg_hess[d_half:, d_half:] = reg_hess[d_half:, d_half:] / scaling
        hess = hess + reg_hess

        # eigh = np.linalg.eigh(hess)
        # print(eigh)

        iter += 1
        update = np.linalg.inv(hess) @ grad
        theta = theta - update
        if np.linalg.norm(update) < 1e-5 or iter > 50:
            break

    # X_data = np.concatenate(offered_contexts, axis=0)

    # cov = X_data.T @ X_data / len(X_data)
    # eigvals = np.linalg.eigvals(cov)
    # print(f"Data Covariance:\n{cov}")

    # clf = LogisticRegression(random_state=0, fit_intercept=False, penalty='l2').fit(
    #     X_data, [0 if np.sum(selected_contexts[tau]) == 0 else 1 for tau in range(len(offered_contexts))])
    # print("clf_coef_ = ", clf.coef_)

    return theta

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
        return solve_assortment_and_pricing(self.K, alpha_star, beta_star)


class DynamicAlgorithms(Algorithm):
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
            assortment, prices = solve_assortment_and_pricing_nonlinear(self.K, h_p, self.L0, self.pool)
        return assortment, prices


class DynamicAssortmentPricing(DynamicAlgorithms):
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
            self.theta = solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta, scaling=self.L0 ** 2)
        else:
            self.V += (1 / self.K ** 2) * x_tilde.T @ x_tilde
        self.t += 1


class NewtonAssortmentPricing(DynamicAlgorithms):

    def __init__(self, n, d, K, L0, T0, pool):
        self.theta0 = np.zeros(2 * d)
        self.alpha_g = 0.001
        super().__init__(n, d, K, L0, T0, pool)

    def mle_online_update(self, offered_contexts, selected_contexts):
        utilities = offered_contexts @ self.theta
        terms = np.exp(utilities)
        probs = terms / (1 + np.sum(terms))
        G = np.sum((probs * offered_contexts.T).T, axis=0) - selected_contexts

        V_half = np.linalg.cholesky(self.V)

        theta_param = cp.Variable(2 * self.d)
        obj = cp.Minimize(cp.sum_squares(theta_param @ V_half) + theta_param @ (0.2 * G - 2 * self.V @ self.theta))
        constraints = [cp.sum_squares(theta_param - self.theta0) <= self.L0]
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
            self.theta = solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta, scaling=self.L0 ** 2 )
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


class JavanmardDynamicPricing(Algorithm):
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
            _, prices = solve_assortment_and_pricing(self.n, alpha, beta)
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
            self.theta = solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta, scaling=self.L0 ** 2)
        self.t += 1
        self.episode_t += 1
        # logging.debug(len(self.selected_contexts))
        if self.episode_t == self.episode_len:
            self.t = 0
            self.episode_t = 0
            self.episode_len += 1


class OhIyengarAssortmentSelection(Algorithm):
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
            self.theta = solve_mle(self.offered_contexts, self.selected_contexts, self.d, init_theta=self.theta)
        self.t += 1


class OhIyengarWithPricing(DynamicAlgorithms):
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
            self.theta = solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta, scaling=self.L0 ** 2)
        self.t += 1


def goyalperivier_update(theta, V, offered_contexts, selected_contexts, lmb):
    utilities = offered_contexts @ theta
    terms = np.exp(utilities)
    probs = terms / (1 + np.sum(terms))
    G = np.sum((probs * offered_contexts.T).T, axis=0) - selected_contexts
    theta = theta - np.linalg.pinv(V + lmb * np.identity(len(V))) @ G
    return theta


class GoyalPerivierDynamicPricing(Algorithm):
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

    def goyalperivier_project(self, contexts):
        V_half = np.linalg.cholesky(self.V)
        theta_param = cp.Variable(2 * self.d)
        obj = cp.Minimize(cp.sum_squares(theta_param @ V_half))
        constraints = [contexts[:, self.d:] @ theta_param[self.d:] >= self.L0]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return np.array(theta_param.value)

    def get_assortment_and_pricing(self, contexts):
        # project the parameter
        self.theta = self.goyalperivier_project(contexts)
        psi, phi = self.theta[:self.d], self.theta[self.d:]
        alpha = contexts[:, :self.d] @ psi
        beta = contexts[:, self.d:] @ phi
        # compute prices assuming all items can be offered
        _, prices = solve_assortment_and_pricing(self.n, alpha, beta)
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
        self.theta = goyalperivier_update(self.theta, self.V, offered_contexts, selected_contexts, 10 * self.d * np.log(self.t + 1))
