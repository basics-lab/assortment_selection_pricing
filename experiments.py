import numpy as np
import uuid
import os
import logging
import time
from multiprocessing import Pool
import argparse
from collections import defaultdict

from assortment_pricing import algorithms

if __name__ == "__main__":

    compute_pool = Pool(5)
    parser = argparse.ArgumentParser(description='Run assortment selection and pricing experiments.')
    parser.add_argument('-d', type=int, help='Dimensionality of the context vectors')
    parser.add_argument('-L0', type=float, help='Minimum price senstitivity')
    parser.add_argument('-T', type=int, help='Number of time periods')
    parser.add_argument('-N', type=int, help='Number of available items')
    parser.add_argument('-K', type=int, help='Number of items in the assortment')
    parser.add_argument('-T0_low', type=int, help='Lower bound for number of initialization periods')

    args = parser.parse_args()

    d = args.d
    L0 = args.L0
    T = args.T
    N = args.N
    K = args.K
    T0_low = args.T0_low

    # d = 5
    # L0 = 0.5
    # T = 2000
    # N = 5
    # K = 5
    # T0_low = 150

    print(f"Arguments passed: d = {d}, L0 = {L0}, T = {T}, K = {K}, N = {N}, T0_low = {T0_low}")

    T0_Dynamic = np.random.randint(T0_low, T0_low + 100, 1)[0] # number of initialization rounds
    T0_Online = int(1.5 * T0_Dynamic) # number of initialization rounds with online learning

    experiment_name = f"d{d}_L{L0}_T{T}_N{N}_K{K}_" + str(uuid.uuid4().hex[:8])
    os.makedirs(f"results/{experiment_name}")
    print(f"Experiment started: {experiment_name}")
    logging.basicConfig(filename=f"results/{experiment_name}/experiment.log", format='%(asctime)s: %(message)s', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    psi_star = np.random.normal(0, 1/np.sqrt(d/2), size=d)
    phi_star = np.random.normal(0, 1/np.sqrt(d/2), size=d)
    theta_star = np.concatenate([psi_star, phi_star])

    logging.info(f"theta_star: {theta_star}")

    def mnl_selection(values):
        items = np.concatenate([np.arange(len(values)), [None]])
        terms = np.concatenate([np.exp(values), [1]])
        probs = terms / np.sum(terms)
        return np.random.choice(items, p=probs)

    def revenue_expectation(values, prices):
        probs = np.exp(values) / (1 + np.sum(np.exp(values)))
        return np.sum(prices * probs)

    def generate_context(n, d):
        context = np.random.normal(0, 1 / np.sqrt(d), size=(n, d))
        context = (np.sign(context @ psi_star) * context.T).T
        context2 = np.random.normal(0, 1 / np.sqrt(d), size=(n, d))
        context2 = (np.sign(context2 @ phi_star) * context2.T).T
        return np.concatenate((context, context2 + L0 * phi_star / np.linalg.norm(phi_star) ** 2), axis=1)

    offlineOptimal = algorithms.OfflineOptimalAlgorithm(K)

    # algorithms
    algorithms_list = [
        ("CAP", algorithms.DynamicAssortmentPricing(N, d, K, L0, T0=T0_Dynamic, pool=compute_pool)),
        ("CAP-ONS", algorithms.NewtonAssortmentPricing(N, d, K, L0, T0=T0_Online, pool=compute_pool)),
        ("M3P", algorithms.JavanmardDynamicPricing(N, d, K, L0)),
        ("DBL-MNL", algorithms.OhIyengarAssortmentSelection(N, d, K, L0, T0=T0_Dynamic, fixed_prices=5)),
        ("DBL-MNL-Pricing", algorithms.OhIyengarWithPricing(N, d, K, L0, T0=T0_Dynamic, pool=compute_pool)),
        ("ONS-MPP", algorithms.GoyalPerivierDynamicPricing(N, d, K, L0))
    ]

    def algorithm_step(
            algo_name: str,
            algo: algorithms.Algorithm,
            contexts,
            alpha_star,
            beta_star,
            optimal_revenue,
    ):
        start = time.time()
        assortment, prices = algo.get_assortment_and_pricing(contexts)
        values = alpha_star - beta_star * prices
        i_t_assortment = mnl_selection(values[assortment])
        i_t = assortment[i_t_assortment] if i_t_assortment is not None else None
        algo.selection_feedback(i_t, contexts, assortment, prices)
        end = time.time()
        observed_revenue = prices[i_t] if i_t is not None else 0
        expected_revenue = revenue_expectation(values[assortment], prices[assortment])
        try:
            theta_NMSE = np.linalg.norm(algo.theta - theta_star) / np.linalg.norm(theta_star)  
        except:
            theta_NMSE = -1
        logging.info(f"{algo_name:<20} theta NMSE: {theta_NMSE:<10.4f} regret = {optimal_revenue - expected_revenue:<10.4f}")
        step_result = {
            "time": end - start,
            "optimal_revenue": optimal_revenue,
            "observed_revenue": observed_revenue,
            "expected_revenue": expected_revenue,
            "theta_NMSE" : theta_NMSE,
        }
        return step_result

    results = defaultdict(list)

    for t in range(T):

        logging.info(f"-------------------------------------")
        logging.info(f"t = {t}, T0_Dynamic = {T0_Dynamic}, T0_Online = {T0_Online}")

        contexts = generate_context(N, d)
        alpha_star = contexts[:, :d] @ psi_star
        beta_star = contexts[:, d:] @ phi_star

        # offline optimum assortment and prices
        assortment, prices = offlineOptimal.get_assortment_and_pricing(alpha_star, beta_star)
        values = alpha_star - beta_star * prices
        i_t = mnl_selection(values[assortment])
        revenue = prices[i_t] if (i_t is not None) else 0
        # logging.info(f"Optimum: {alpha_star[assortment]}, {beta_star[assortment]}, {prices[assortment]}, {np.exp(values[assortment]) / (1 + np.sum(np.exp(values[assortment])))}")
        optimal_revenue = revenue_expectation(values[assortment], prices[assortment])
        logging.info(f"Optimum expected revenue = {optimal_revenue}")

        for algo_key, algo in algorithms_list:
            step_result = algorithm_step(algo_key, algo, contexts, alpha_star, beta_star, optimal_revenue)
            step_result.update({
                "t": t
            })
            results[algo_key].append(step_result)
