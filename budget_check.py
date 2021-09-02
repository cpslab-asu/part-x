def budget_check(remaining_budget, options, Nc, Nc_, n_subrem):
    """
    :param total_budget: Maximum Number of Evaluations (T)
    :param initialization_budget: Budget for sampling points
    :param number_of_BO_samples: Budget for BO samples
    :param continued_sampling_budget: Budget for Continious Sampling
    :param Nc: Minimum Evaluations per subregion for Classified
    :param Nc_: Minimum Evaluations per subregion for Unclassified
    :param n_subrem: Number of regions with class 'remaining' (r,r+,r-)
    :return: Total Budget remaining after the iteration
    """

    total_budget = remaining_budget
    initialization_budget = options.initialization_budget
    number_of_BO_samples = options.number_of_BO_samples[0]
    continued_sampling_budget = options.continued_sampling_budget
    budget_for_iter =  max((Nc*n_subrem)-(number_of_BO_samples+initialization_budget),0)+max((Nc_)-(continued_sampling_budget),0)
    print("Budget for iteration = {}".format(budget_for_iter))
    if total_budget >= budget_for_iter:
        
        ##Execute BO, Quatile Estimations,Classification
        total_budget = total_budget - (number_of_BO_samples+initialization_budget)
        ##Allocate continued_sampling_budget to the classified subregions(+,-) propotionally
        total_budget = total_budget - (continued_sampling_budget)
        return True
    else:
        ##Allocate subregions proprotionally to the volume
        ##Evaluate the function at the sampled point and update the Gaussian processes
        return False