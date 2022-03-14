"""
    Model Data
"""

struct ModelData{T,X,U,W}
    dynamics::Vector{Dynamics{T}}
    jacobian_state::Vector{X}
    jacobian_action::Vector{U}
	jacobian_parameter::Vector{W}
end

function model_data(model::Model)
	jacobian_state = [zeros(d.ny, d.nx) for d in model]
    jacobian_action = [zeros(d.ny, d.nu) for d in model]
	jacobian_parameter = [zeros(d.ny, d.nw) for d in model]
    ModelData(model, jacobian_state, jacobian_action, jacobian_parameter)
end

function reset!(data::ModelData) 
    T = length(data.jacobian_state) + 1
    for t = 1:T-1 
        fill!(data.jacobian_state[t], 0.0) 
        fill!(data.jacobian_action[t], 0.0) 
        # fill!(data.jacobian_parameter[t], 0.0) 
    end 
end