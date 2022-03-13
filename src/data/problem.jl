mutable struct ProblemData{T,X,U,D,C,FX,FU,FW,OX,OU,OXX,OUU,OUX}
    # current trajectory
    x::Vector{X}
    u::Vector{U}

    # disturbance trajectory
    w::Vector{D}

    # nominal trajectory
    x̄::Vector{X}
    ū::Vector{U}

    # dynamics model
    model::ModelData{T,FX,FU,FW}

    # objective derivatives data
    objective::ObjectiveData{C,OX,OU,OXX,OUU,OUX}

    # z = (x1...,xT,u1,...,uT-1) | Δz = (Δx1...,ΔxT,Δu1,...,ΔuT-1)
    z::Vector{T}
end

function problem_data(model::Model, obj; 
    w=[[zeros(d.nw) for d in model]..., zeros(0)])

    length(w) == length(model) && (w = [w..., zeros(0)])
    @assert length(model) + 1 == length(w)
    @assert length(model) + 1 == length(obj)

	x = [[zeros(d.nx) for d in model]..., 
            zeros(model[end].ny)]
    u = [[zeros(d.nu) for d in model]..., zeros(0)]

    x̄ = [[zeros(d.nx) for d in model]..., 
            zeros(model[end].ny)]
    ū = [[zeros(d.nu) for d in model]..., zeros(0)]

    md = model_data(model)
    od = objective_data(model, obj)

    z = zeros(num_var(model))

    ProblemData(x, u, w, x̄, ū, md, od, z)
end