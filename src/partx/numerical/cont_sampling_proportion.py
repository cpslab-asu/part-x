import numpy as np

def assign_budgets(vol_probablity_distribution, continued_sampling_budget):

    cumu_sum = np.cumsum(np.insert(vol_probablity_distribution, 0,0))
    # print("Cumulative_sum list = {}".format(cumu_sum))
    random_numbers = np.random.uniform(0.0,1.0, continued_sampling_budget)
    n_cont_budget_distribution = []
    for iterate in range(len(cumu_sum)-1):
        bool_array = np.logical_and(random_numbers > cumu_sum[iterate], random_numbers <= cumu_sum[iterate+1])
        n_cont_budget_distribution.append(bool_array.sum())
    # print(n_cont_budget_distribution)
    return (n_cont_budget_distribution)


# n_cont = 10
# a = [0.,0.,0.,0.]

# print(assign_budgets(a,n_cont))

