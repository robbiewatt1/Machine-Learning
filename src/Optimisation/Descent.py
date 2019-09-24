import numpy as np
from autograd import grad

# This module contains some optimisation methods to be used when gradients can
# be easily calculated. Enjoy!


class Descent:
    """
    Base class for module optimisation classes.
    """
    def __init__(self, func, params, epsilon, iters, rate, auto_grad=True):
        """
        :param func         Functions to be optimised.
        :param params:      Numpy array of initial guess of parameters being
                            optimised.
        :param rate:        Rate of descent
        :param iters:       Number of iterations before breaking.
        :param epsilon:     Minimum change in params of convergence.
        :param auto_grad    Switch for using auto gradient
        """

        self.func = func
        self.params = params
        self.n_params = len(self.params)
        self.rate = rate
        self.iters = iters
        self.epsilon = epsilon

        if auto_grad:
            # We use auto gradient package
            self.jac = grad(func)
        else:
            # We use finite difference function
            self.jac = self._jacobian
            # Small number for finite difference
            self.fd_epsilon = 1.0e-10

    def optimise(self):
        pass

    def _jacobian(self, x):
        """
        2 point finite difference calculation of jacobian
        :param x: (Numpy array) input location
        :return:  (Numpy array) Jacobian of func, evaluated at x.
        """
        # Might be a faster way using matrices
        f1 = self.func(x)
        jac = np.zeros([self.n_params])
        for i in range(self.n_params):
            x2 = np.copy(x)
            x2[i] += self.fd_epsilon
            f2 = self.func(x2)
            jac[i] = (f2 - f1) / self.fd_epsilon
        return jac

    def _line_search(self, dir, x, alpha0=1., c=1.0e-4):
        """
        Method for preforming an inexact line search using Armijo backtracking.
        I used Scipy minimise for help.

        :param dir:   Direction of line search
        :param x      Current location
        :param alpha0 Initial guess for step length
        :param c      Armijo condition rule.
        :return: step_length
        """

        # Scalar function along dir
        def func_scalar(alpha):
            return self.func(x + alpha * dir)

        # Derivative of scalar function along dir
        def der_func_scalar(alpha):
            return np.dot(self.jac(x + alpha * dir), dir)

        # Function and derivative evaluated at 0
        func0 = func_scalar(0.)
        der_func0 = der_func_scalar(0.)
        func_a0 = func_scalar(alpha0)

        # Return if step is good
        if func_a0 <= func0 + c * alpha0 * der_func0:
            return alpha0

        alpha1 = -der_func0 * alpha0 ** 2. / 2.0 / (func_a0 - func0 - der_func0 * alpha0)
        func_a1 = func_scalar(alpha1)

        # Return if step is good now
        if func_a1 <= func0 + c * alpha0 * der_func0:
            return alpha1

        # loop with cubic interpolation until alpha
        # satisfies the first Wolfe condition
        while alpha1 > 0.:  # we are assuming alpha>0 is a descent direction
            factor = alpha0 ** 2. * alpha1 ** 2. * (alpha1 - alpha0)
            a = alpha0 ** 2. * (func_a1 - func0 - der_func0 * alpha1) - \
                alpha1 ** 2. * (func_a0 - func0 - der_func0 * alpha0)
            a = a / factor
            b = -alpha0 ** 3. * (func_a1 - func0 - der_func0 * alpha1) + \
                alpha1 ** 3. * (func_a0 - func0 - der_func0 * alpha0)
            b = b / factor

            alpha2 = (-b + np.sqrt(abs(b ** 2. - 3. * a * der_func0))) / (3.0 * a)
            func_a2 = func_scalar(alpha2)

            if func_a2 <= func0 + c * alpha2 * der_func0:
                return alpha2

            if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
                alpha2 = alpha1 / 2.0

            alpha0 = alpha1
            alpha1 = alpha2
            func_a0 = func_a1
            func_a1 = func_a2

        # Failed to find a suitable step length
        print("Warning: Line searched failed, returning 1.")
        return 1.0


class CoordinateDescent(Descent):
    """
    Coordinate descent optimiser for a given function.
    """
    def __init__(self, func, params, jac=None, rate=0.01, epsilon=0.0001, iters=1000):
        """
        :param func         Functions to be optimised.
        :param params:      Numpy array of initial guess of parameters being
                            optimised.
        :param jac:         Jacobian for func. If None then a finite difference
                            method will be used.
        :param rate:        Rate of descent
        :param iters:       Number of iterations before breaking.
        :param epsilon:     Minimum change in params of convergence.
        """
        super(CoordinateDescent, self).__init__(func, params, jac, epsilon, iters, rate)

    def optimise(self):
        """
        Method to minimise using coordinate decent
        :return: minimum of function
        """
        # loop over iterations
        for step in range(self.iters):
            grad = self.jac(self.params)

            # loop through parameters
            for i in range(self.n_params):
                # find local gradient
                self.params[i] -= self.jac(self.params)[i] * self.rate

            # check for convergence
            if np.linalg.norm(grad) < self.epsilon:
                break
            elif step == self.iters-1:
                print("Warning: optimisation has not converged")

        return self.params


class GradientDescent(Descent):
    """
    Gradient descent optimiser for a given function.

    """
    def __init__(self, func, params, jac=None, epsilon=0.00001, iters=10000, rate=0.01,):
        """
        :param func         Functions to be optimised.
        :param params:      Numpy array of initial guess of parameters being
                            optimised.
        :param jac:         Jacobian for func. If None then a finite difference
                            method will be used.
        :param rate:        Rate of descent
        :param iters:       Number of iterations before breaking.
        :param epsilon:     Minimum change in params of convergence.
        """
        super(GradientDescent, self).__init__(func, params, jac, epsilon, iters, rate)

    def optimise(self):
        """
        Method to minimise using coordinate decent
        :return: minimum of function
        """
        # loop over iterations
        for step in range(self.iters):
            grad = self.jac(self.params)
            self.params -= grad * self.rate

            # check for convergence
            if np.linalg.norm(grad) < self.epsilon:
                break
            elif step == self.iters-1:
                print("Warning: optimisation has not converged")

        return self.params


class BFGS(Descent):
    """
    BFGS optimiser for a given function. Has O(n^2) convergence rate
    """

    def __init__(self, func, params, jac=None, epsilon=0.00001, iters=10000):
        """
        :param func         Functions to be optimised.
        :param params:      Numpy array of initial guess of parameters being
                            optimised.
        :param jac:         Jacobian for func. If None then a finite difference
                            method will be used.
        :param iters:       Number of iterations before breaking.
        :param epsilon:     Minimum change in params of convergence.
        """
        super(BFGS, self).__init__(func, params, jac, epsilon, iters, rate=None)

    def optimise(self):
        """
        Method to minimise using BFGS decent.
        :return: Numpy array of optimised paramters
        """

        # Initialise the inverse of the hessian approximation as identity
        hess_inv = np.identity(self.n_params)
        # Find the jacobian at current location

        for step in range(self.iters):
            # Solve for descent direction
            grad_old = self.jac(self.params)
            descent_dir = -1. * hess_inv @ grad_old

            # Preform line search for step length
            s = self._line_search(descent_dir, self.params) * descent_dir

            # Update parameter and check convergence
            self.params += s
            if np.linalg.norm(self.jac(self.params)) < self.epsilon:
                break
            elif step == self.iters-1:
                print("Warning: Optimisation has not converged")

            # Find change in gradient and update hessian
            grad_new = self.jac(self.params)
            y = grad_new - grad_old
            sy = np.inner(s, y)
            hess_inv += (sy + np.inner(y, hess_inv @ y)) \
                * np.outer(s, s) / sy**2. \
                - (hess_inv @ np.outer(y, s)
                   + np.outer(s, (np.transpose(y) @ hess_inv))) / sy

        return self.params
