"""
    Model Data
"""

struct ModelData{T,X,U,W}
    dynamics::Vector{Dynamics{T}}
    jacobian_state::Vector{X}
    jacobian_action::Vector{U}
	jacobian_parameter::Vector{W}
end

function model_data(dynamics::Vector{Dynamics{T}}) where T
	jacobian_state = [zeros(d.ny, d.nx) for d in dynamics]
    jacobian_action = [zeros(d.ny, d.nu) for d in dynamics]
	jacobian_parameter = [zeros(d.ny, d.nw) for d in dynamics]
    ModelData(model, jacobian_state, jacobian_action, jacobian_parameter)
end

function reset!(data::ModelData) 
    H = length(data.dynamics) + 1
    for t = 1:H-1 
        fill!(data.jacobian_state[t], 0.0) 
        fill!(data.jacobian_action[t], 0.0) 
        fill!(data.jacobian_parameter[t], 0.0) 
    end 
end 