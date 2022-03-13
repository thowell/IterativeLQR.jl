

mutable struct ProblemData{T,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
    # current trajectory
    x::Vector{X}
    u::Vector{U}

    # disturbance trajectory
    w::Vector{D}

    # nominal trajectory
    x̄::Vector{X}
    ū::Vector{U}

    # model data
    model::ModelData{T,FX,FU,FW}

    # objective data
    objective::ObjectiveData{O,OX,OU,OXX,OUU,OUX}

    # z = (x1...,xT,u1,...,uT-1) | Δz = (Δx1...,ΔxT,Δu1,...,ΔuT-1)
    z::Vector{T}
end

function problem_data(dynamics, costs; 
    w=[[zeros(d.nw) for d in dynamics]..., zeros(0)])

    length(w) == length(dynamics) && (w = [w..., zeros(0)])
    @assert length(dynamics) + 1 == length(w)
    @assert length(dynamics) + 1 == length(costs)

	x = [[zeros(d.nx) for d in dynamics]..., 
            zeros(dynamics[end].ny)]
    u = [[zeros(d.nu) for d in dynamics]..., zeros(0)]

    x̄ = [[zeros(d.nx) for d in dynamics]..., 
            zeros(dynamics[end].ny)]
    ū = [[zeros(d.nu) for d in dynamics]..., zeros(0)]

    model = model_data(dynamics)
    obj = objective_data(dynamics, costs)

    z = zeros(num_var(dynamics))

    ProblemData(x, u, w, x̄, ū, model, obj, z)
end