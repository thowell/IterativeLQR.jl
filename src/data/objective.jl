

struct ObjectiveData{C,X,U,XX,UU,UX}
    costs::C
    gradient_state::Vector{X}
    gradient_action::Vector{U}
    hessian_state_state::Vector{XX}
    hessian_action_action::Vector{UU}
    hessian_action_state::Vector{UX}
end

function objective_data(model::Model, obj)
	gradient_state = [[zeros(d.nx) for d in model]..., 
        zeros(model[end].ny)]
    gradient_action = [zeros(d.nu) for d in model]
    hessian_state_state = [[zeros(d.nx, d.nx) for d in model]..., 
        zeros(model[end].ny, model[end].ny)]
    hessian_action_action = [zeros(d.nu, d.nu) for d in model]
    hessian_action_state = [zeros(d.nu, d.nx) for d in model]
    ObjectiveData(obj, gradient_state, gradient_action, hessian_state_state, hessian_action_action, hessian_action_state)
end

function reset!(data::ObjectiveData) 
    T = length(data.gradient_state) 
    for t = 1:T 
        fill!(data.gradient_state[t], 0.0) 
        fill!(data.hessian_state_state[t], 0.0) 
        t == T && continue
        fill!(data.gradient_action[t], 0.0)
        fill!(data.hessian_action_action[t], 0.0)
        fill!(data.hessian_action_state[t], 0.0)
    end 
end