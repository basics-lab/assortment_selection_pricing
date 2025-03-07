import numpy as np
import uuid
import os
import logging
import time
from multiprocessing import Pool
import argparse
from collections import defaultdict
import pandas as pd

from assortment_pricing import algorithms, baselines

if __name__ == "__main__":

    compute_pool = Pool(5)
    parser = argparse.ArgumentParser(description='Run assortment selection and pricing experiments.')
    parser.add_argument('-d', type=int, help='Dimensionality of the context vectors')
    parser.add_argument('-L0', type=float, help='Minimum price senstitivity')
    parser.add_argument('-T', type=int, help='Number of time periods')
    parser.add_argument('-N', type=int, help='Number of available items')
    parser.add_argument('-K', type=int, help='Number of items in the assortment')
    parser.add_argument('-T0_low', type=int, help='Lower bound for number of initialization periods')
    parser.add_argument('-result_dir', type=str, help='Directory to save results')

    args = parser.parse_args()

    d = args.d
    L0 = args.L0
    T = args.T
    N = args.N
    K = args.K
    result_dir = args.result_dir

    hyperparams = {
        "N": args.N,
        "K": args.K,
        "d": args.d,
        "L0": args.L0,
        "T": args.T,
    }

    print(f"Arguments passed: d = {d}, L0 = {L0}, T = {T}, K = {K}, N = {N}")

    T0_low = int(np.sqrt(T))
    T0_Dynamic = np.random.randint(T0_low, 2 * T0_low, 1)[0] # number of initialization rounds
    T0_Online = T0_Dynamic # number of initialization rounds with online learning

    experiment_dir = os.path.join(result_dir, str(uuid.uuid4().hex[:8]))
    os.makedirs(experiment_dir)
    print(f"Experiment started: {experiment_dir}")
    log_file_path = os.path.join(experiment_dir, "experiment.log")
    logging.basicConfig(filename=log_file_path, format='%(asctime)s: %(message)s', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    psi_star = np.random.normal(0, 1/np.sqrt(d/2), size=d)
    phi_star = np.random.normal(0, 1/np.sqrt(d/2), size=d)
    theta_star = np.concatenate([psi_star, phi_star])

    logging.info(f"theta_star: {theta_star}")

    def mnl_selection(values):
        items = np.concatenate([np.arange(len(values)), [None]])
        logits = np.concatenate((values, [0]))
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
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
        ("M3P", baselines.JavanmardDynamicPricing(N, d, K, L0)),
        ("DBL-MNL", baselines.OhIyengarAssortmentSelection(N, d, K, L0, T0=T0_Dynamic, fixed_prices=5)),
        ("DBL-MNL-Pricing", baselines.OhIyengarWithPricing(N, d, K, L0, T0=T0_Dynamic, pool=compute_pool)),
        ("ONS-MPP", baselines.GoyalPerivierDynamicPricing(N, d, K, L0)),
        ("Thompson", baselines.GaussianThompsonSampling(N, d, K, L0, T0=T0_Dynamic, pool=compute_pool))
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

    results = []

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
                "algo": algo_key,
                "t": t,
            })
            step_result.update(hyperparams)
            results.append(step_result)

results_df = pd.DataFrame(results)
filename = os.path.join(experiment_dir, "results.parquet")
results_df.to_parquet(filename, index=False)
print(f"Results saved to {filename}.")
