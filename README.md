# IterativeLQR.jl
[![CI](https://github.com/thowell/IterativeLQR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/thowell/IterativeLQR.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/thowell/IterativeLQR.jl/branch/main/graph/badge.svg?token=FGM33O1K1E)](https://codecov.io/gh/thowell/IterativeLQR.jl)

A Julia package for solving constrained trajectory optimization problems with iterative LQR (iLQR). 

$$ 
\begin{align*}
		\underset{x_{1:T}, \phantom{\,} u_{1:T-1}}{\text{minimize }} & \phantom{,} g_T(x_T; \theta_T) + \sum \limits_{t = 1}^{T-1} g_t(x_t, u_t; \theta_t)\\
		\text{subject to } & \phantom{,} f_t(x_t, u_t; \theta_t) = x_{t+1}, \phantom{,} \quad t = 1,\dots,T-1,\\
		& \phantom{,} c_t(x_t, u_t; \theta_t) \phantom{,}[\leq, =] \phantom{,} 0, \quad t = 1, \dots, T,\\
\end{align*}
$$

with

- $x_{1:T}$: state trajectory 
- $u_{1:T-1}$: action trajectory 
- $\theta_{1:T}$: problem-data trajectory 


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

function particle_discrete(x, u)
   A = [1.0 1.0; 0.0 1.0]
   B = [0.0; 1.0] 
   return A * x + B * u[1]
end

# model
particle = Dynamics(particle_discrete, num_state, num_action)
model = [particle for t = 1:T-1] 

# initialization
x1 = [0.0; 0.0] 
xT = [1.0; 0.0]
ū = [1.0e-1 * randn(num_action) for t = 1:T-1] 
x̄ = rollout(model, x1, ū)

# objective  
objective = [
   [Cost((x, u) -> 0.1 * dot(x, x) + 0.1 * dot(u, u), num_state, num_action) for t = 1:T-1]..., 
   Cost((x, u) -> 0.1 * dot(x, x), num_state, 0)
]

# constraints
constraints = [
   [Constraint() for t = 1:T-1]..., 
   Constraint((x, u) -> x - xT, num_state, 0)
] 

# solver
solver = Solver(model, objective, constraints)
initialize_controls!(solver, ū)
initialize_states!(solver, x̄)

# solve
solve!(solver)

# solution
x_sol, u_sol = get_trajectory(solver)
```
## Examples 

Please see the following for examples using this package: 

- [Trajectory Optimization with Optimization-Based Dynamics](https://github.com/thowell/optimization_dynamics) 
- [Dojo: A Differentiable Simulator for Robotics](https://github.com/dojo-sim/Dojo.jl)
