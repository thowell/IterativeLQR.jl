

struct ObjectiveData{C,X,U,XX,UU,UX}
    costs::C
    gradient_state::Vector{X}
    gradient_action::Vector{U}
    hessian_state_state::Vector{XX}
    hessian_action_action::Vector{UU}
    hessian_action_state::Vector{UX}
end

function objective_data(dynamics::Vector{Dynamics{T}}, costs) where T
	gradient_state = [[zeros(d.num_state) for d in dynamics]..., 
        zeros(dynamics[end].num_next_state)]
    gradient_action = [zeros(d.num_action) for d in dynamics]
    hessian_state_state = [[zeros(d.num_state, d.num_state) for d in dynamics]..., 
        zeros(dynamics[end].num_next_state, dynamics[end].num_next_state)]
    hessian_action_action = [zeros(d.num_action, d.num_action) for d in dynamics]
    hessian_action_state = [zeros(d.num_action, d.num_state) for d in dynamics]
    ObjectiveData(costs, gradient_state, gradient_action, hessian_state_state, hessian_action_action, hessian_action_state)
end

function reset!(data::ObjectiveData) 
    H = length(data.gradient_state) 
    for t = 1:H 
        fill!(data.gradient_state[t], 0.0) 
        fill!(data.hessian_state_state[t], 0.0) 
        t == H && continue
        fill!(data.gradient_action[t], 0.0)
        fill!(data.hessian_action_action[t], 0.0)
        fill!(data.hessian_action_state[t], 0.0)
    end 
end