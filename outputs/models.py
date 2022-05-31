class SmoothSVM(object):

    def __init__(self, h=None, max_iter=10000, eta=1e-4, fit_intercept=True, step_size=0.01,
                 lambda_penalty=0, theta_init="random"):
        '''
        If h = None, optimal h 9from Bahadur remainder) persepctive is determined once .fit() method is called
        '''
        self.h = h
        self.eta = eta
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.step_size = step_size
        self.lambda_penalty = lambda_penalty
        self.theta_init = theta_init
        # self.random_state = random_state

    def fit(self, X, Y):
        """Fit training data.
          ----------
          X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of
          samples and
          n_features is the number of features.
          y : array-like, shape = [n_samples]
          Target values.
          Returns
          -------
          self : object
        """
        if self.fit_intercept == True:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        else:
            pass

        self.converged = False
        self.convergence_count = 0

        # set initial value for theta based on arguments provided
        if self.theta_init == "random":
            theta = np.random.randn(p + 1) / 10
        else:
            theta = self.theta_init

        for i in range(self.max_iter):

            self.gradient = self._calculate_conv_loss_gradient(Y, X, theta)  # evalueate gradient at current theta

            theta_new = theta - self.step_size * self.gradient  # perform gradient descent (single step)

            if np.linalg.norm(theta - theta_new) < self.eta:
                self.converged = True
                self.convergence_count += 1
                break

            theta = theta_new  # update theta

            self.coefs = theta

        return (self)

    def _calculate_kvals(self, Y, X, theta):
        '''Calculate value of smoothed loss function for a list of x values and given bandwidth h.

        Inputs:
        - h: bandwidth (float)
        - x: feature value (if vector, calculate loss for each vector)

        Output:
        - kernel_loss (vector of the same simension as input "x")
        '''
        # note that quad integrates along the first axis (along "u" in the calc_conv_integrand function).
        fitted_val = X @ theta  # get N x 1 vector
        errors = np.ones(len(Y)) - Y * fitted_val  # elementwise multiplication, get N x 1 vector
        K_vals = norm.cdf(errors / self.h)  # get N x 1 vector of K_values

        return (K_vals)

    def _calculate_conv_loss_gradient(self, Y, X, theta):
        '''Calculate gradient of the convolutional loss given the values for h, Y, X and theta.

        Inputs:
        - h: bandwidth (float)
        - Y: Nx1 vector
        - X: Nxp np.array
        - theta: px1 np.array - parameter vecotr to evaluate gradient at

        Outputs:
        - grad_vector (np.array of the same shape as theta): gradient of the conv smoothed loss at a given param. value theta.
        '''

        K_vals = self._calculate_kvals(Y, X, theta)

        # Note that reshaping below implicitly transposes the vectors
        yx_product = Y.reshape(-1, 1) * X  # get N x p matrix, in each row is y_{i} @ x_{i}, 1 x p vector
        grad_i_vector = K_vals.reshape(-1,
                                       1) * yx_product  # result is a N x p matrix, which we need to sum over N along each column
        theta_penalty = theta.copy()
        theta_penalty[0] = 0  # don't penalise the intercept/bias
        # grad_vector = - 1/len(Y) * grad_i_vector.sum(axis = 0) + lambda_penalty * theta_penalty # sum along each columns, get p x 1 gradient vector
        grad_vector = - 1 / len(Y) * grad_i_vector.sum(axis=0) + self.lambda_penalty * theta_penalty  # - 100*1/200

        return (grad_vector)


class HingeSVM(object):

    def __init__(self):
        pass

    def calculate_sample_gradient(self, X, Y):
        X2 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        indicator = 1 - Y * (X2 @ self.coefs)
        gradient = (X2.T * Y.T).T
        gradient = gradient[indicator >= 0].mean(axis=0)

        self.sample_gradient = gradient

        return (self)

    def fit(self, X, Y):
        '''Mimnimise hinge loss with no regularisation (lambda = 0) for given data X and Y using linear programming (simplex method).

        i.e. min \sum_{i=1}^{n} ( 1 - y_{i} < x_{i}, \theta > )_{+}

        Inputs:
        - X: dataframe of features (excluding constant from bias)
        - Y: dataframe of response variable (Y_{i} takes values -1, 1)

        Outputs:
        - coefs: numpy array of the model coefficients
        - res: result object from the linear program (can be used to check slack variables, convergence, etc).
        '''
        n, p = X.shape[0], X.shape[1]

        obj_coefs_epsilon = np.ones(n)  # coefs for slack variables
        obj_coefs_x = np.zeros(p + 1)  # coefs for theta ( theta = (b, w) )
        obj_coefs = np.concatenate((obj_coefs_epsilon, obj_coefs_x))

        # Constrants to ensure non-negatuve slack variables
        # Left hand side
        ub_lhs_positivity_constraints = np.concatenate((np.eye(n), np.zeros((n, p + 1))), axis=1)
        # RHS
        ub_rhs_positivity_constraints = np.zeros(n)

        # Constraints for residual: slack_i >= 1 - y_{i} < x_{i}, \theta >
        ub_lhs_residual_epsilon = np.eye(n)  # one equation per slack variable
        ub_lhs_residual_x = ((X.T) * (Y.T)).T  # coefficients of w (weight vector)
        ub_lhs_residual_bias = np.ones(n) * Y  # coefficients for bias (b)
        ub_lhs_residual_xbias = np.concatenate((ub_lhs_residual_bias.reshape(-1, 1), ub_lhs_residual_x),
                                               axis=1)  # combine coefs for theta
        # combine with slack coefs to get coefs for the left-hand side of the inequality constraint
        ub_lhs_residual = np.concatenate((ub_lhs_residual_epsilon, ub_lhs_residual_xbias),
                                         axis=1)  # combine coefs for slack vars and theta
        # RHS of the inequality constraint
        ub_rhs_residual = np.ones(n)

        # reverse signs as Python uses "<=" constraint whereas we need ">=".
        lhs_coefs = - np.concatenate((ub_lhs_positivity_constraints, ub_lhs_residual), axis=0)
        rhs_coefs = - np.concatenate((ub_rhs_positivity_constraints, ub_rhs_residual), axis=0)

        # Solve the linear program
        res = linprog(obj_coefs,  # specify objective function
                      A_ub=lhs_coefs, b_ub=rhs_coefs,  # specify constraint ("<=")
                      bounds=(None, None),  # no bounds on theta
                      method='revised simplex')  # simplex method

        # extract coefficients corresponfing to theta (i.e. ignore those corresp to slack vars)
        coefs = res.x[-(p + 1):]

        self.converged = res.success

        self.coefs = coefs

        return (self)

class PopulationSVM(object):

  def __init__(self):
    pass

  def fit(self, mu1, mu2, Sigma):
    # store set-up parameters in an object
    self.params = dict(mu1 =  mu1, mu2 = mu2, Sigma =  Sigma)

    d = self._get_mahalanobis_distance(mu1, mu2, Sigma)
    a = self._get_gamma_inverse(d/2)
    # a = a/2
    Sigma_inv = np.linalg.inv(Sigma)

    diff_means = (mu1 - mu2).reshape(-1,1)
    sum_means = (mu1 + mu2).reshape(-1,1)

    bias = -(diff_means.T @ Sigma_inv @ sum_means)  / (2*a*d + (d**2))
    w = (2 * Sigma_inv @ diff_means) / (2*a*d + (d**2))

    coefs = np.concatenate((bias.reshape(1, 1), w))
    self.coefs = coefs

    # set some private attributes to avoid re-calculating them
    self._a = a
    self._mah_dist = d

    return( self )

  @staticmethod
  def _get_mahalanobis_distance(mu1, mu2, Sigma):

    mu_diff = mu1 - mu2
    Sigma_inv = np.linalg.inv(Sigma)

    return( (mu_diff.T @ Sigma_inv @ mu_diff)**(1/2) )

  @staticmethod
  def _gamma_function(a):
    gamma = norm.pdf(a) / norm.cdf(a)

    return(gamma)

  @staticmethod
  def _get_gamma_derivative(x):
    term1 = -x*norm.pdf(x)/norm.cdf(x)
    term2 = - norm.pdf(x)**2 / norm.cdf(x)**2
    return( term1 + term2 )

  def _get_gamma_inverse(self, a):
    x = 0

    for i in range(1000):
      x = x -  (self._gamma_function(x) -a)/ self._get_gamma_derivative(x)

    return( x )

  def get_correction_constant(self, mu_delta_growth = None ):

    if mu_delta_growth == None:
      mu_delta_growth = np.sqrt( len(self.mu1) )

    alpha = 0.2
    c = mu_delta_growth

    for i in range(1000):

      der_gamma_argument = (2*self._a + self.mah_dist - self._mah_dist*c) / ( 2*np.sqrt(alpha) )

      term_1_der = - 0.5* self._get_gamma_derivative(der_gamma_argument) * ( 2*self._a + self._mah_dist - self._mah_dist*c ) / ( 2 * c * alpha )

      term_2_der = 1/2 * (1/np.sqrt(alpha)) * (1/c) * self._gamma_function(der_gamma_argument)

      function_value = self._gamma_function( der_gamma_argument ) * (np.sqrt(alpha)/c)

      alpha = alpha -  (function_value - d/2)/ (term_1_der + term_2_der)

    return( alpha )

  def get_gradient(self):

    mu1 = self.params["mu1"].reshape(-1,1)
    mu2 = self.params["mu2"].reshape(-1,1)
    Sigma = self.params["Sigma"]

    d = self._mah_dist
    a = self._a

    G22 = mu1 @ mu1.T + mu2 @ mu2.T + 2*Sigma - (a/d + 1) * (mu1-mu2) @ (mu1 - mu2).T

    G11 = np.array([2]).reshape(1,1)
    G12 = (mu1 + mu2).T
    G21 = G12.T

    G_const = norm.cdf(a) / 2

    G1 = np.concatenate((G11, G12), axis = 1)
    G2 = np.concatenate((G21.reshape(-1,1), G22), axis = 1)

    G = np.concatenate((G1.reshape(1, -1), G2), axis = 0)
    G = G_const * G

    return( G )


  def get_hessian(self):

      mu1 = self.params["mu1"].reshape(-1,1)
      mu2 = self.params["mu2"].reshape(-1,1)
      Sigma = self.params["Sigma"]

      d = self._mah_dist
      a = self._a


      H22 = mu1 @ mu1.T + mu2 @ mu2.T + 2*Sigma + 2*( (a/d)**2 + a/d - 1/(d**2) ) * (mu1 - mu2) @ (mu1 - mu2).T

      H11 = np.array([2]).reshape(1,1)
      H12 = (mu1 + mu2).T
      H21 = H12.T

      H_const = norm.pdf(a)/4 * (2*a + d)

      H1 = np.concatenate((H11, H12), axis = 1)
      H2 = np.concatenate((H21.reshape(-1,1), H22), axis = 1)

      H = np.concatenate((H1.reshape(1, -1), H2), axis = 0)
      H = H_const * H

      # check positive definiteness (Cholesky used for comp efficiency)
      try:
        np.linalg.cholesky(H)
      except:
        raise ValueError("The Hessian matrix is not positive definite.")

      return( H )



