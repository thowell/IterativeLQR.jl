[![CI](https://github.com/thowell/IterativeLQR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/thowell/IterativeLQR.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/thowell/IterativeLQR.jl/branch/main/graph/badge.svg?token=FGM33O1K1E)](https://codecov.io/gh/thowell/IterativeLQR.jl)
# IterativeLQR.jl
A Julia package for solving constrained trajectory optimization problems with iterative LQR (iLQR). 

```
minimize        gT(xT; wT) + sum(gt(xt, ut; wt))
xt, ut
subject to      xt+1 = ft(xt, ut; wt) , t = 1,...,T-1 
                x1 = x_init
                ct(xt, ut; wt) {<,=} 0, t = 1,...,T
```

Fast and allocation-free gradients and Jacobians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs, constraints, and dynamics. Constraints are handled using an augmented Lagrangian framework. 

## Installation
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add https://github.com/thowell/IterativeLQR.jl
```