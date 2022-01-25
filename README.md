# IterativeLQR.jl
[![CI](https://github.com/thowell/IterativeLQR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/thowell/IterativeLQR.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/thowell/IterativeLQR.jl/branch/main/graph/badge.svg?token=FGM33O1K1E)](https://codecov.io/gh/thowell/IterativeLQR.jl)
A Julia package for solving constrained trajectory optimization problems with iterative LQR (iLQR). 

```
minimize        gT(xT; wT) + sum(gt(xt, ut; wt))
x1:T, u1:T-1
subject to      xt+1 = ft(xt, ut; wt) , t = 1,...,T-1 
                x1 = x_init
                ct(xt, ut; wt) {<,=} 0, t = 1,...,T
```

Fast and allocation-free gradients and Jacobians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs, constraints, and dynamics. Constraints are handled using an augmented Lagrangian framework. 

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
nx = 2
nu = 1 

function particle(x, u, w)
   A = [1.0 1.0; 0.0 1.0]
   B = [0.0; 1.0] 
   return A * x + B * u[1]
end

# model
dyn = Dynamics(particle, nx, nu)
model = [dyn for t = 1:T-1] 

# initialization
x1 = [0.0; 0.0] 
xT = [1.0; 0.0]
ū = [1.0e-1 * randn(nu) for t = 1:T-1] 
x̄ = rollout(model, x1, ū)

# objective 
ot = (x, u, w) -> 0.1 * dot(x, x) + 0.1 * dot(u, u)
oT = (x, u, w) -> 0.1 * dot(x, x)
ct = Cost(ot, nx, nu)
cT = Cost(oT, nx, 0)
obj = [[ct for t = 1:T-1]..., cT]

# constraints
goal(x, u, w) = x - xT

cont = Constraint()
conT = Constraint(goal, nx, 0)
cons = [[cont for t = 1:T-1]..., conT] 

# problem
prob = problem_data(model, obj, cons)
initialize_controls!(prob, ū)
initialize_states!(prob, x̄)

# solve
solve!(prob, verbose=true)

# solution
x_sol, u_sol = get_trajectory(prob)
```
