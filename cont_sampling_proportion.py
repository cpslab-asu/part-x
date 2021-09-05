import numpy as np

def assign_budgets(vol_probablity_distribution, continued_sampling_budget):
    cumu_sum = np.cumsum([0] + vol_probablity_distribution)
    random_numbers = np.random.uniform(0.0,1.0, continued_sampling_budget)
    n_cont_budget_distribution = []
    for iterate in range(len(cumu_sum)-1):
        bool_array = np.logical_and(random_numbers > cumu_sum[iterate], random_numbers <= cumu_sum[iterate+1])
        n_cont_budget_distribution.append(bool_array.sum())
    return (n_cont_budget_distribution)

# n_cont = 10
# # a = [0.1, 0.3, 0.4, 0.15, 0.05]
# a = [0.5, 0.25,0.25]



# print(assign_budgets(a,n_cont))

