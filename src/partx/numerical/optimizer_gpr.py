from scipy.optimize import fmin_l_bfgs_b

def optimizer_lbfgs_b(obj_func, initial_theta, bounds):
    # * 'obj_func': the objective function to be minimized, which
    #   takes the hyperparameters theta as a parameter and an
    #   optional flag eval_gradient, which determines if the
    #   gradient is returned additionally to the function value
    # * 'initial_theta': the initial value for theta, which can be
    #   used by local optimizers
    # * 'bounds': the bounds on the values of theta


    # options = {'maxiter' : 1e10, 'maxfun':1e10}
    # print(initial_theta.reshape(1,initial_theta.shape[0]).shape)
    # params = minimize(obj_func, initial_theta.reshape(1,initial_theta.shape[0]), method = 'l-bfgs-b',bounds = None, options = options)

    params = fmin_l_bfgs_b(obj_func, initial_theta,bounds = None, maxiter = 30000, maxfun=1e10)
    # print(params[0][0], params[1])
    # Returned are the best found hyperparameters theta and
    # the corresponding value of the target function.
    return params[0], params[1]