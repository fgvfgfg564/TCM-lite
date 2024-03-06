from scipy.optimize import minimize


def objective_function(x):
    return x[0] ** 2 + x[1] ** 2  # Example objective function: minimize sum of squares


def gradient(x):
    return [2 * x[0], 2 * x[1]]  # Gradient of the objective function


# Define bounds for each variable
bounds = [(-1, 1), (-1, 1)]  # Example bounds: each variable between -1 and 1


# Define inequality constraint: sum of variables <= 1
def inequality_constraint(x):
    return x[0] + x[1] - 0.25


constraint = {"type": "ineq", "fun": inequality_constraint}

# Initial guess for the parameters
x0 = [-5, -5]

# Minimize the objective function using L-BFGS-B algorithm with constraints
result = minimize(
    objective_function,
    x0,
    jac=gradient,
    method="SLSQP",
    bounds=bounds,
    constraints=[constraint],
)


# Print the result
print("Optimal parameters:", result.x)
print("Optimal objective value:", result.fun)
