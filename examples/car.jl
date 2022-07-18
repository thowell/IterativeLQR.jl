# PREAMBLE

# PKG_SETUP

# ## Setup

using IterativeLQR 
using LinearAlgebra
using Plots

# ## horizon 
T = 51 

# ## car 
num_state = 3
num_action = 2
num_parameter = 0 

function car_continuous(x, u)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function car_discrete(x, u)
    h = 0.1 # timestep 
    x + h * car_continuous(x + 0.5 * h * car_continuous(x, u), u)
end

# ## model
car = Dynamics(car_discrete, num_state, num_action)
dynamics = [car for t = 1:T-1] 

# ## initialization
x1 = [0.0; 0.0; 0.0] 
xT = [1.0; 1.0; 0.0] 

# ## rollout
ū = [1.0e-2 * [1.0; 0.1] for t = 1:T-1]
x̄ = rollout(dynamics, x1, ū)

# ## objective 
objective = [
    [Cost((x, u) -> 1.0 * dot(x - xT, x - xT) + 1.0e-2 * dot(u, u), num_state, num_action) for t = 1:T-1]...,
    Cost((x, u) -> 1000.0 * dot(x - xT, x - xT), num_state, 0)
]

# ## constraints
ul = -5.0 * ones(num_action) 
uu = 5.0 * ones(num_action)

p_obs = [0.5; 0.5] 
r_obs = 0.1

constraints = [
    [Constraint((x, u) -> begin
        e = x[1:2] - p_obs
        [
            ul - u; ## control limit (lower)
            u - uu; ## control limit (upper)
            r_obs^2.0 - dot(e, e); ## obstacle 
        ]
    end, 
    num_state, num_action, indices_inequality=collect(1:5)) for t = 1:T-1]..., 
    Constraint((x, u) -> begin
        e = x[1:2] - p_obs
        [
            x - xT; # goal 
            r_obs^2.0 - dot(e, e); # obstacle
        ]
    end, num_state, num_action, indices_inequality=collect(3 .+ (1:1)))
] 

# ## solver
solver = Solver(dynamics, objective, constraints)
initialize_controls!(solver, ū) 
initialize_states!(solver, x̄)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

# ## visualize
plot(hcat(x_sol...)')
plot(hcat(u_sol[1:end-1]...)', linetype=:steppost)

# ## benchmark allocations + timing
# using BenchmarkTools
# info = @benchmark solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))
