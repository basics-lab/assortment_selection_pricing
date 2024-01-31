import numpy as np
import uuid
import os
import algorithms
import logging
import time
from multiprocessing import Pool


if __name__ == "__main__":

    compute_pool = Pool(8)

    d = 5
    L0 = 0.2
    T = 1000
    N = 100
    K = 5

    T0_Dynamic = np.random.randint(50, 60, 1)

    experiment_name = f"d{d}_T{T}_n{N}_K{K}_" + str(uuid.uuid4().hex[:8])
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


    def expected_revenue(values, prices):
        probs = np.exp(values) / (1 + np.sum(np.exp(values)))
        return np.sum(prices * probs)


    def generate_context(n, d):
        context = np.random.normal(0, 1 / np.sqrt(d), size=(n, d))
        context = (np.sign(context @ psi_star) * context.T).T
        context2 = np.random.normal(0, 1 / np.sqrt(d), size=(n, d))
        context2 = (np.sign(context2 @ phi_star) * context2.T).T
        return np.concatenate((context, context2 + L0 * phi_star / np.linalg.norm(phi_star) ** 2), axis=1)


    # algorithms
    offlineOptimal = algorithms.OfflineOptimalAlgorithm(K)
    dynamicAssortment = algorithms.DynamicAssortmentPricing(N, d, K, L0, T0=T0_Dynamic, pool=compute_pool)
    newtonAssortment = algorithms.NewtonAssortmentPricing(N, d, K, L0, T0=T0_Dynamic, pool=compute_pool)
    javanmardPricing = algorithms.JavanmardDynamicPricing(N, d, K, L0)
    ohIyengarAssortment = algorithms.OhIyengarAssortmentSelection(N, d, K, L0, T0=T0_Dynamic, fixed_prices=5)
    ohIyengarPricing = algorithms.OhIyengarWithPricing(N, d, K, L0, T0=T0_Dynamic, pool=compute_pool)
    goyalPerivierDynamicPricing = algorithms.GoyalPerivierDynamicPricing(N, d, K, L0)

    # logger
    history_expected_revenue = np.zeros((7, T))
    history_time = np.zeros((6, T))

    for t in range(T):

        logging.info(f"-------------------------------------")
        logging.info(f"t = {t}")

        contexts = generate_context(N, d)
        alpha_star = contexts[:, :d] @ psi_star
        beta_star = contexts[:, d:] @ phi_star

        # print(beta_star)

        # offline optimum algo
        assortment, prices = offlineOptimal.get_assortment_and_pricing(alpha_star, beta_star)
        values = alpha_star - beta_star * prices
        i_t = mnl_selection(values[assortment])
        revenue = prices[i_t] if (i_t is not None) else 0

        logging.info(f"Optimum: {alpha_star[assortment]}, {beta_star[assortment]}, {prices[assortment]}, {np.exp(values[assortment]) / (1 + np.sum(np.exp(values[assortment])))}")

        history_expected_revenue[0, t] = expected_revenue(values[assortment], prices[assortment])
        logging.info(f"Optimum expected revenue = {history_expected_revenue[0, t]}")

        # our algorithm (MLE from scratch)
        start = time.time()
        assortment, prices = dynamicAssortment.get_assortment_and_pricing(contexts)
        values = alpha_star - beta_star * prices
        i_t_assortment = mnl_selection(values[assortment])
        i_t = assortment[i_t_assortment] if i_t_assortment is not None else None
        dynamicAssortment.selection_feedback(i_t, contexts, assortment, prices)
        revenue = prices[i_t] if i_t is not None else 0
        end = time.time()

        # logging.info(f"{alpha_star[assortment]}, {beta_star[assortment]}, {prices[assortment]}, {np.exp(values[assortment]) / (1 + np.sum(np.exp(values[assortment])))}, {i_t_assortment}")
        # logging.info(f"Dynamic Assortment theta NMSE: {np.linalg.norm(dynamicAssortment.theta - theta_star) / np.linalg.norm(theta_star)}")
        # logging.info(dynamicAssortment.theta)
        history_expected_revenue[1, t] = expected_revenue(values[assortment], prices[assortment])
        history_time[0, t] = end - start
        logging.info(f"Dynamic Assortment regret = {history_expected_revenue[0, t] - history_expected_revenue[1, t]}")

        # our algorithm (online parameter update)
        start = time.time()
        assortment, prices = newtonAssortment.get_assortment_and_pricing(contexts)
        values = alpha_star - beta_star * prices
        i_t_assortment = mnl_selection(values[assortment])
        i_t = assortment[i_t_assortment] if i_t_assortment is not None else None
        newtonAssortment.selection_feedback(i_t, contexts, assortment, prices)
        revenue = prices[i_t] if i_t is not None else 0
        end = time.time()

        # logging.info(f"Newton Assortment theta NMSE: {np.linalg.norm(newtonAssortment.theta - theta_star) / np.linalg.norm(theta_star)}")
        history_expected_revenue[2, t] = expected_revenue(values[assortment], prices[assortment])
        history_time[1, t] = end - start
        logging.info(f"Newton Assortment regret = {history_expected_revenue[0, t] - history_expected_revenue[2, t]}")

        # Javanmard (pricing only)
        start = time.time()
        assortment, prices = javanmardPricing.get_assortment_and_pricing(contexts)
        values = alpha_star - beta_star * prices
        i_t_assortment = mnl_selection(values[assortment])
        i_t = assortment[i_t_assortment] if i_t_assortment is not None else None
        javanmardPricing.selection_feedback(i_t, contexts, assortment, prices)
        revenue = prices[i_t] if i_t is not None else 0
        end = time.time()

        # logging.info(f"Javanmard Pricing theta NMSE: {np.linalg.norm(javanmardPricing.theta - theta_star) / np.linalg.norm(theta_star)}")
        history_expected_revenue[3, t] = expected_revenue(values[assortment], prices[assortment])
        history_time[2, t] = end - start
        logging.info(f"Javanmard Pricing regret = {history_expected_revenue[0, t] - history_expected_revenue[3, t]}")

        # Oh & Iyengar (assortment selection only)
        start = time.time()
        assortment, prices = ohIyengarAssortment.get_assortment_and_pricing(contexts)
        values = alpha_star - beta_star * prices
        i_t_assortment = mnl_selection(values[assortment])
        i_t = assortment[i_t_assortment] if i_t_assortment is not None else None
        ohIyengarAssortment.selection_feedback(i_t, contexts, assortment, prices)
        revenue = prices[i_t] if i_t is not None else 0
        end = time.time()

        # logging.info(f"Oh & Iyengar Pricing theta NMSE: {np.linalg.norm(ohIyengarPricing.theta - theta_star) / np.linalg.norm(theta_star)}")
        history_expected_revenue[4, t] = expected_revenue(values[assortment], prices[assortment])
        history_time[3, t] = end - start
        logging.info(f"Oh & Iyengar Assortment regret = {history_expected_revenue[0, t] - history_expected_revenue[4, t]}")

        # Oh & Iyengar with added pricing
        start = time.time()
        assortment, prices = ohIyengarPricing.get_assortment_and_pricing(contexts)
        values = alpha_star - beta_star * prices
        i_t_assortment = mnl_selection(values[assortment])
        i_t = assortment[i_t_assortment] if i_t_assortment is not None else None
        ohIyengarPricing.selection_feedback(i_t, contexts, assortment, prices)
        revenue = prices[i_t] if i_t is not None else 0
        end = time.time()

        # logging.info(f"{alpha_star[assortment]}, {beta_star[assortment]}, {prices[assortment]}, {np.exp(values[assortment]) / (1 + np.sum(np.exp(values[assortment])))}, {i_t_assortment}")
        # logging.info(f"Oh & Iyengar with Pricing theta NMSE: {np.linalg.norm(ohIyengarPricing.theta - theta_star) / np.linalg.norm(theta_star)}")
        # logging.info(ohIyengarPricing.theta)
        history_expected_revenue[5, t] = expected_revenue(values[assortment], prices[assortment])
        history_time[4, t] = end - start
        logging.info(f"Oh & Iyengar with Pricing regret = {history_expected_revenue[0, t] - history_expected_revenue[5, t]}")

        # Goyal & Perivier for dynamic pricing (no assortment selection)
        start = time.time()
        assortment, prices = goyalPerivierDynamicPricing.get_assortment_and_pricing(contexts)
        values = alpha_star - beta_star * prices
        i_t_assortment = mnl_selection(values[assortment])
        i_t = assortment[i_t_assortment] if i_t_assortment is not None else None
        goyalPerivierDynamicPricing.selection_feedback(i_t, contexts, assortment, prices)
        revenue = prices[i_t] if i_t is not None else 0
        end = time.time()

        # logging.info(f"{alpha_star[assortment]}, {beta_star[assortment]}, {prices[assortment]}, {np.exp(values[assortment]) / (1 + np.sum(np.exp(values[assortment])))}, {i_t_assortment}")
        logging.info(f"Goyal & Perivier Pricing theta NMSE: {np.linalg.norm(goyalPerivierDynamicPricing.theta - theta_star) / np.linalg.norm(theta_star)}")
        # logging.info(ohIyengarPricing.theta)
        history_expected_revenue[6, t] = expected_revenue(values[assortment], prices[assortment])
        history_time[5, t] = end - start
        logging.info(
            f"Goyal & Perivier Pricing regret = {history_expected_revenue[0, t] - history_expected_revenue[6, t]}")

        total_revenue = np.sum(history_expected_revenue, axis=1)
        logging.info(total_revenue[0] - total_revenue[1:])

    np.save(f"results/{experiment_name}/history_expected_revenue.npy", history_expected_revenue)
    np.save(f"results/{experiment_name}/history_time.npy", history_time)
