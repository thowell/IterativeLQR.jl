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
dyn = Dynamics(midpoint_explicit, nx, nu, nw)
model = [dyn for t = 1:T-1] 
data_model = model_derivatives_data(model)

# ## initialization
x1 = [0.0; 0.0; 0.0] 
xT = [1.0; 1.0; 0.0] 

# ## rollout
ū = [1.0e-2 * [1.0; 0.1] for t = 1:T-1]
w = [zeros(nw) for t = 1:T-1] 
x̄ = rollout(model, x1, ū, w)

plot(hcat(x̄...)')

# ## objective 
ot = (x, u, w) -> 1.0 * dot(x - xT, x - xT) + 1.0e-2 * dot(u, u)
oT = (x, u, w) -> 1000.0 * dot(x - xT, x - xT)
ct = Cost(ot, nx, nu, nw)
cT = Cost(oT, nx, 0, nw)
obj = [[ct for t = 1:T-1]..., cT]

prob = problem_data(model, obj)
initialize_control!(prob, ū) 
# initialize_state!(prob, x̄)
# prob.s_data.α[1]
# rollout!(prob.p_data, prob.m_data, α=prob.s_data.α[1])
# plot(hcat(prob.m_data.x...)')

# prob.p_data.K

ilqr_solve!(prob, 
    obj_tol=1.0e-3,
    grad_tol=1.0e-3,
    max_iter=100)

x_sol, u_sol = nominal_trajectory(prob)

plot(hcat(x_sol...)')

# ## constraints
# function ctrl_bnd(x, u, w) 
#     return [u - ul; uu - u]
# end
# ctrlt = StageConstraint(ctrl_bnd, nx, nu, nw, [t for t = 1:T-1], :inequality)

# function goal(x, u, w) 
#     return x - xT
# end
# goalT = StageConstraint(goal, nx, nu, nw, [T], :equality)

# p_obs = [0.5; 0.5] 
# r_obs = 0.1
# function obs(x, u, w) 
#     e = x[1:2] - p_obs
#     return [dot(e, e) - r_obs^2.0]
# end
# obst = StageConstraint(obs, nx, nu, nw, [t for t = 1:T-1], :inequality)
# obsT = StageConstraint(obs, nx, 0, nw, [T], :inequality)

# cons = ConstraintSet([obst, obsT, ctrlt, goalT])

ul = -2.0 * ones(nu) 
uu = 2.0 * ones(nu)

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

con_empty = Constraint()
cont = Constraint(stage_con, nx, nu, idx_ineq=collect(1:5))
conT = Constraint(terminal_con, nx, nu, idx_ineq=collect(3 .+ (1:1)))
cons = [[cont for t = 1:T-1]..., conT] 
# cons = [[con_empty for t = 1:T-1]..., conT] 

# obj_al = augmented_lagrangian(model, obj, cons)
# prob = problem_data(model, obj)
prob = problem_data(model, obj, cons)
initialize_control!(prob, ū) 
initialize_state!(prob, x̄)
constraints!(prob.m_data.obj.c_data, prob.m_data.x̄, prob.m_data.ū, prob.m_data.w)
active_set!(prob.m_data.obj.a, prob.m_data.obj.c_data, prob.m_data.obj.λ)
prob.m_data.obj.c_data.c[1]
prob.m_data.obj.a

prob.m_data.obj.c_data.c[4]

prob.m_data.obj.c_data.c[1]
prob.m_data.obj.c_data.cons[1].idx_ineq
prob.m_data.obj.c_data.cons[2].idx_ineq
sum((prob.m_data.x̄[1][1:2] - p_obs).^2.0) - r_obs^2.0 
prob.m_data.obj.a[end]
# constraints!(obj_al.c_data, prob.m_data.x, prob.m_data.u, prob.m_data.w)
# active_set!(obj_al.a, obj_al.c_data, obj_al.λ)

# eval_obj(obj_al, prob.m_data.x, prob.m_data.u, prob.m_data.w)
# augmented_lagrangian_update!(obj_al)
# prob.m_data.obj
# objective_derivatives!(prob.m_data.obj, prob.m_data, mode=:nominal)

constrained_ilqr_solve!(prob, 
    linesearch=:armijo,
    α_min=1.0e-5,
    obj_tol=1.0e-3,
    grad_tol=1.0e-3,
    max_iter=100,
    max_al_iter=10,
    ρ_init=1.0,
    ρ_scale=10.0)

x_sol, u_sol = nominal_trajectory(prob)

plot(hcat(x_sol...)')
plot(hcat(u_sol...)', linetype=:steppost)

