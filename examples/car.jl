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

function car(x, u, w)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function midpoint_explicit(x, u, w)
    h = 0.1 # timestep 
    x + h * car(x + 0.5 * h * car(x, u, w), u, w)
end

# ## model
dynamics = Dynamics(midpoint_explicit, num_state, num_action, num_parameter)
model = [dynamics for t = 1:T-1] 

# ## initialization
x1 = [0.0; 0.0; 0.0] 
xT = [1.0; 1.0; 0.0] 

# ## rollout
ū = [1.0e-2 * [1.0; 0.1] for t = 1:T-1]
w = [zeros(num_parameter) for t = 1:T] 
x̄ = rollout(model, x1, ū, w)

# ## objective 
ot = (x, u, w) -> 1.0 * dot(x - xT, x - xT) + 1.0e-2 * dot(u, u)
oT = (x, u, w) -> 1000.0 * dot(x - xT, x - xT)
ct = Cost(ot, num_state, num_action, num_parameter)
cT = Cost(oT, num_state, 0, num_parameter)
objective = [[ct for t = 1:T-1]..., cT]

# ## constraints
ul = -5.0 * ones(num_action) 
uu = 5.0 * ones(num_action)

p_obs = [0.5; 0.5] 
r_obs = 0.1

function stage_con(x, u, w) 
    e = x[1:2] - p_obs
    [
    ul - u; # control limit (lower)
    u - uu; # control limit (upper)
    r_obs^2.0 - dot(e, e); # obstacle 
    ]
end 

function terminal_con(x, u, w) 
    e = x[1:2] - p_obs
    [
    x - xT; # goal 
    r_obs^2.0 - dot(e, e); # obstacle
    ]
end

cont = Constraint(stage_con, num_state, num_action, indices_inequality=collect(1:5))
conT = Constraint(terminal_con, num_state, num_action, indices_inequality=collect(3 .+ (1:1)))
constraints = [[cont for t = 1:T-1]..., conT] 

# ## problem
prob = Solver(model, objective, constraints)
initialize_controls!(prob, ū) 
initialize_states!(prob, x̄)

# ## solve
solve!(prob)

# ## solution
x_sol, u_sol = get_trajectory(prob)

# ## visualize
plot(hcat(x_sol...)')
plot(hcat(u_sol[1:end-1]...)', linetype=:steppost)

# ## benchmark allocations + timing
using BenchmarkTools
info = @benchmark solve!($prob, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))
