using Test
using Symbolics
using ForwardDiff
using LinearAlgebra
using IterativeLQR

include("objective.jl")
include("dynamics.jl")
include("constraints.jl")

include("acrobot.jl")
include("car.jl")