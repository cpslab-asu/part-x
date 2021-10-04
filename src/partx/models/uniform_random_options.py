class uniform_random_options:
    def __init__(self, start_seed, number_of_samples, initial_region_support, test_function_dimension, BENCHMARK_NAME):
                
        self.start_seed = start_seed
        self.initial_region_support = initial_region_support
        self.test_function_dimension = test_function_dimension 
        # self.number_of_samples_gen_GP = number_of_samples_gen_GP
        self.number_of_samples = number_of_samples
        self.BENCHMARK_NAME = BENCHMARK_NAME
