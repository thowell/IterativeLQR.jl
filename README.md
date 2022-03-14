# IterativeLQR.jl
[![CI](https://github.com/thowell/IterativeLQR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/thowell/IterativeLQR.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/thowell/IterativeLQR.jl/branch/main/graph/badge.svg?token=FGM33O1K1E)](https://codecov.io/gh/thowell/IterativeLQR.jl)

A Julia package for solving constrained trajectory optimization problems with iterative LQR (iLQR). 

```
minimize        cost_T(state_T; parameter_T) + sum(cost_t(state_t, action_t; parameter_t))
states, actions
subject to      state_t+1 = dynamics_t(state_t, action_t; parameter_t), t = 1,...,T-1 
                state_1 = state_initial
                constraint_t(state_t, action_t; parameter_t) {<,=} 0,   t = 1,...,T
```


- Fast and allocation-free gradients and Jacobians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs, constraints, and dynamics. 

- Constraints are handled using an [augmented Lagrangian framework](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method). 

- Cost, dynamics, and constraints can have varying dimensions at each time step.

- Parameters are exposed (and gradients wrt these values coming soon!)

For more details, see our related paper: [ALTRO: A Fast Solver for Constrained Trajectory Optimization](http://roboticexplorationlab.org/papers/altro-iros.pdf)

## Installation
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add https://github.com/thowell/IterativeLQR.jl
```

## Quick Start 
```julia
using IterativeLQR 
using LinearAlgebra

# horizon 
T = 11 

# particle 
num_state = 2
num_action = 1 

function particle(x, u, w)
   A = [1.0 1.0; 0.0 1.0]
   B = [0.0; 1.0] 
   return A * x + B * u[1]
end

# model
dyn = Dynamics(particle, num_state, num_action)
model = [dyn for t = 1:T-1] 

# initialization
x1 = [0.0; 0.0] 
xT = [1.0; 0.0]
ū = [1.0e-1 * randn(num_action) for t = 1:T-1] 
x̄ = rollout(model, x1, ū)

# objective 
ot = (x, u, w) -> 0.1 * dot(x, x) + 0.1 * dot(u, u)
oT = (x, u, w) -> 0.1 * dot(x, x)
ct = Cost(ot, num_state, num_action)
cT = Cost(oT, num_state, 0)
obj = [[ct for t = 1:T-1]..., cT]

# constraints
goal(x, u, w) = x - xT

cont = Constraint()
conT = Constraint(goal, num_state, 0)
cons = [[cont for t = 1:T-1]..., conT] 

# problem
prob = solver(model, obj, cons)
initialize_controls!(prob, ū)
initialize_states!(prob, x̄)

# solve
solve!(prob)

# solution
x_sol, u_sol = get_trajectory(prob)
```
## Examples 

Please see the following for examples using this package: 

- [Trajectory Optimization with Optimization-Based Dynamics](https://github.com/thowell/optimization_dynamics) 
- [Dojo: A Differentiable Simulator for Robotics](https://github.com/dojo-sim/Dojo.jl)
