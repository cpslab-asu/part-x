def budget_check(options, function_calls, remaining_nodes):
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

    max_budget = options.max_budget
    budget_exhausted = function_calls
    budget_available = max_budget - budget_exhausted

    initialization_budget = options.initialization_budget
    number_of_BO_samples = options.number_of_BO_samples[0]
    continued_sampling_budget = options.continued_sampling_budget

    remaining_regions_count = len(remaining_nodes)*options.test_function_dimension

    # classified_region_count = len(classified_nodes)
    budget_for_iter =  remaining_regions_count * (initialization_budget + number_of_BO_samples) + continued_sampling_budget
    # print("*******************************************************************")
    # print("Budget Available = {}".format(budget_available))
    # print("Estimated Budget for iteration = {}".format(budget_for_iter))
    if budget_available >= budget_for_iter:
        # print("Going ahead with normal flow")
        # print("*******************************************************************")
        return True
    else:
        # print("breaking loop and going for the last act")
        # print("*******************************************************************")
        return False