# PREAMBLE

# PKG_SETUP

# ## Setup

using IterativeLQR 
using LinearAlgebra
using Plots

# ## horizon 
T = 51 

# ## car 
nx = 3
nu = 2
nw = 0 

function car(x, u, w)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function midpoint_explicit(x, u, w)
    h = 0.1 # timestep 
    x + h * car(x + 0.5 * h * car(x, u, w), u, w)
end

# ## model
dt = Dynamics(midpoint_explicit, nx, nu, nw=nw)
dyn = [dt for t = 1:T-1] 
model = DynamicsModel(dyn)
data_model = dynamics_derivatives_data(model)

# ## initialization
x1 = [0.0; 0.0; 0.0] 
xT = [1.0; 1.0; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.0 * dot(x - xT, x - xT) + 1.0 * dot(u, u)
oT = (x, u, w) -> 0.0 * dot(x - xT, x - xT)
ct = Cost(ot, nx, nu, nw, [t for t = 1:T-1])
cT = Cost(oT, nx, 0, nw, [T])
obj = [ct, cT]
objective_derivatives_data(model)

model_data(model, obj)
policy_data(model)
solver_data(model)


# ## constraints
ul = -0.5 * ones(nu) 
uu = 0.5 * ones(nu)

function ctrl_bnd(x, u, w) 
    return [u - ul; uu - u]
end
ctrlt = StageConstraint(ctrl_bnd, nx, nu, nw, [t for t = 1:T-1], :inequality)

function goal(x, u, w) 
    return x - xT
end
goalT = StageConstraint(goal, nx, nu, nw, [T], :equality)

p_obs = [0.5; 0.5] 
r_obs = 0.1
function obs(x, u, w) 
    e = x[1:2] - p_obs
    return [dot(e, e) - r_obs^2.0]
end
obst = StageConstraint(obs, nx, nu, nw, [t for t = 1:T-1], :inequality)
obsT = StageConstraint(obs, nx, 0, nw, [T], :inequality)

cons = ConstraintSet([obst, obsT, ctrlt, goalT])