Base.@kwdef mutable struct Options{T} 
    linesearch::Symbol=:armijo
    max_iter::Int=100
    max_al_iter::Int=10
    α_min::T=1.0e-5
    obj_tol::T=1.0e-3
    grad_tol::T=1.0e-3
    con_tol::T=5.0e-3
    con_norm_type::T=Inf
    ρ_init::T=1.0
    ρ_scale::T=10.0
    ρ_max::T=1.0e8
    reset_cache::Bool=false
    verbose=true
end