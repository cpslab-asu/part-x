def budget_check(options, function_calls, remaining_nodes):
    """[summary]Check if the budget is remaining to continue or move to Continued Sampling phase

    Args:
        options ([type]): Initialization Options
        function_calls ([type]): number of function calls made
        remaining_nodes ([type]): remaining regions

    Returns:
        [type]: If False, Move to continued sampling
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
    if (budget_available >= budget_for_iter) and remaining_regions_count!=0:
        # print("Going ahead with normal flow")
        # print("*******************************************************************")
        return True
    else:
        # print("breaking loop and going for the last act")
        # print("*******************************************************************")
        return False