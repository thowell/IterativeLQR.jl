"""
    Model Data
"""

struct ModelData{T,X,U,W}
    dynamics::Vector{Dynamics{T}}
    fx::Vector{X}
    fu::Vector{U}
	fw::Vector{W}
end

function model_data(model::Model)
	fx = [zeros(d.ny, d.nx) for d in model]
    fu = [zeros(d.ny, d.nu) for d in model]
	fw = [zeros(d.ny, d.nw) for d in model]
    ModelData(model, fx, fu, fw)
end

function reset!(data::ModelData) 
    T = length(data.fx) + 1
    for t = 1:T-1 
        fill!(data.fx[t], 0.0) 
        fill!(data.fu[t], 0.0) 
        # fill!(data.fw[t], 0.0) 
    end 
end