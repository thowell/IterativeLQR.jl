# PREAMBLE

# PKG_SETUP

# ## Setup

using IterativeLQR 
using LinearAlgebra

# ## horizon 
T = 11 

# ## acrobot 
num_state = 2
num_action = 1 

function particle_discrete(x, u)
   A = [1.0 1.0; 0.0 1.0]
   B = [0.0; 1.0] 
   return A * x + B * u[1]
end

# ## model
particle = Dynamics(particle_discrete, num_state, num_action)
dynamics = [particle for t = 1:T-1] 

# ## initialization
x1 = [0.0; 0.0] 
xT = [1.0; 0.0]
ū = [1.0e-1 * randn(num_action) for t = 1:T-1] 
x̄ = rollout(dynamics, x1, ū)

# ## objective 
objective = [
   [Cost((x, u) -> 0.1 * dot(x, x) + 0.1 * dot(u, u), num_state, num_action) for t = 1:T-1]...,
   Cost((x, u) -> 0.1 * dot(x, x), num_state, 0)
]

# ## constraints
constraints = [
   [Constraint() for t = 1:T-1]..., 
   Constraint((x, u) -> x - xT, num_state, 0)
] 

# ## solver
solver = Solver(dynamics, objective, constraints)
initialize_controls!(solver, ū)
initialize_states!(solver, x̄)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)
