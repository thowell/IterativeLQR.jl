struct Cost{T}
    #TODO: types for methods
    evaluate
    gradient_state 
    gradient_action
    hessian_state_state 
    hessian_action_action 
    hessian_action_state
    evaluate_cache::Vector{T}
    gradient_state_cache::Vector{T}
    gradient_action_cache::Vector{T}
    hessian_state_state_cache::Matrix{T}
    hessian_action_action_cache::Matrix{T}
    hessian_action_state_cache::Matrix{T}
end

function Cost(f::Function, num_state::Int, num_action::Int; num_parameter::Int=0)
    #TODO: option to load/save methods
    @variables x[1:num_state], u[1:num_action], w[1:num_parameter]
    
    evaluate = f(x, u, w)
    gradient_state = Symbolics.gradient(evaluate, x)
    gradient_action = Symbolics.gradient(evaluate, u) 
    hessian_state_state = Symbolics.jacobian(gradient_state, x) 
    hessian_action_action = Symbolics.jacobian(gradient_action, u) 
    hessian_action_state = Symbolics.jacobian(gradient_action, x) 

    evaluate_func = eval(Symbolics.build_function([evaluate], x, u, w)[2])
    gradient_state_func = eval(Symbolics.build_function(gradient_state, x, u, w)[2])
    gradient_action_func = eval(Symbolics.build_function(gradient_action, x, u, w)[2])
    hessian_state_state_func = eval(Symbolics.build_function(hessian_state_state, x, u, w)[2])
    hessian_action_action_func = eval(Symbolics.build_function(hessian_action_action, x, u, w)[2])
    hessian_action_state_func = eval(Symbolics.build_function(hessian_action_state, x, u, w)[2])  

    return Cost(evaluate_func, 
        gradient_state_func, gradient_action_func, 
        hessian_state_state_func, hessian_action_action_func, hessian_action_state_func,
        zeros(1), 
        zeros(num_state), zeros(num_action), 
        zeros(num_state, num_state), zeros(num_action, num_action), zeros(num_action, num_state))
end

Objective{T} = Vector{Cost{T}} where T

function cost(costs::Vector{Cost{T}}, states, actions, parameters) where T
    J = 0.0
    for (t, cost) in enumerate(costs)
        cost.evaluate(cost.evaluate_cache, states[t], actions[t], parameters[t])
        J += cost.evaluate_cache[1]
    end
    return J 
end

function cost_gradient!(gradient_states, gradient_actions, costs::Vector{Cost{T}}, states, actions, parameters) where T
    H = length(costs)
    for (t, cost) in enumerate(costs)
        cost.gradient_state(cost.gradient_state_cache, states[t], actions[t], parameters[t])
        @views gradient_states[t] .= cost.gradient_state_cache
        fill!(cost.gradient_state_cache, 0.0) # TODO: confirm this is necessary
        t == H && continue
        cost.gradient_action(cost.gradient_action_cache, states[t], actions[t], parameters[t])
        @views gradient_actions[t] .= cost.gradient_action_cache
        fill!(cost.gradient_action_cache, 0.0) # TODO: confirm this is necessary
    end
end

function cost_hessian!(hessian_state_state, hessian_action_action, hessian_action_state, costs::Vector{Cost{T}}, states, actions, parameters) where T
    H = length(costs) 
    for (t, cost) in enumerate(costs)
        cost.hessian_state_state(cost.hessian_state_state_cache, states[t], actions[t], parameters[t])
        @views hessian_state_state[t] .+= cost.hessian_state_state_cache
        fill!(cost.hessian_state_state_cache, 0.0) # TODO: confirm this is necessary
        t == H && continue
        cost.hessian_action_action(cost.hessian_action_action_cache, states[t], actions[t], parameters[t])
        cost.hessian_action_state(cost.hessian_action_state_cache, states[t], actions[t], parameters[t])
        @views hessian_action_action[t] .+= cost.hessian_action_action_cache
        @views hessian_action_state[t]  .+= cost.hessian_action_state_cache
        fill!(cost.hessian_action_action_cache, 0.0) # TODO: confirm this is necessary
        fill!(cost.hessian_action_state_cache, 0.0) # TODO: confirm this is necessary
    end
end