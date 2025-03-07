import itertools
import numpy as np
from functools import reduce
from scipy import optimize as opt


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
            utilities = np.concat((utilities, [0]))
            terms = np.exp(utilities - np.max(utilities))
            probs = terms / np.sum(terms)
            probs = probs[:-1]
            grad_new = probs @ offered_contexts - selected_contexts
            W = np.diag(probs) - np.outer(probs, probs)
            hess_new = offered_contexts.T @ W @ offered_contexts
            return grad_new, hess_new

        results = map(grad_map, zip(offered_contexts, selected_contexts))
        grad, hess = reduce(grad_reduce, results)

        lmb = 5

        reg_hess = lmb * np.identity(d)
        reg_hess[d_half:, d_half:] = reg_hess[d_half:, d_half:] / scaling
        hess = hess + reg_hess

        iter += 1
        update = np.linalg.inv(hess) @ grad
        theta = theta - update
        if np.linalg.norm(update) < 1e-5 or iter > 50:
            break

    return theta