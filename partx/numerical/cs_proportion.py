import numpy as np

def assign_budgets(vol_probablity_distribution, cs_budget):

    cumu_sum = np.cumsum(np.insert(vol_probablity_distribution, 0,0))
    random_numbers = np.random.uniform(0.0, 1.0, cs_budget)
    n_cont_budget_distribution = []
    for iterate in range(len(cumu_sum)-1):
        bool_array = np.logical_and(random_numbers > cumu_sum[iterate], random_numbers <= cumu_sum[iterate+1])
        n_cont_budget_distribution.append(bool_array.sum())
    return n_cont_budget_distribution
