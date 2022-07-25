struct Dynamics{T}
    evaluate
    jacobian_state
    jacobian_action
    num_next_state::Int
    num_state::Int
    num_action::Int
    num_parameter::Int
    evaluate_cache::Vector{T}
    jacobian_state_cache::Matrix{T}
    jacobian_action_cache::Matrix{T}
end

Model{T} = Vector{Dynamics{T}} where T

function Dynamics(f::Function, num_state::Int, num_action::Int; num_parameter::Int=0)
    #TODO: option to load/save methods
    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_action)
    w = Symbolics.variables(:w, 1:num_parameter)
    # @variables x[1:num_state], u[1:num_action], w[1:num_parameter]

    y = num_parameter > 0 ? f(x, u, w) : f(x, u)
    jacobian_state = Symbolics.jacobian(y, x);
    jacobian_action = Symbolics.jacobian(y, u);
    evaluate_func = eval(Symbolics.build_function(y, x, u, w)[2]);
    jacobian_state_func = eval(Symbolics.build_function(jacobian_state, x, u, w)[2]);
    jacobian_action_func = eval(Symbolics.build_function(jacobian_action, x, u, w)[2]);
    num_next_state = length(y)

    return Dynamics(evaluate_func, jacobian_state_func, jacobian_action_func,
                    num_next_state, num_state, num_action, num_parameter,
                    zeros(num_next_state), zeros(num_next_state, num_state), zeros(num_next_state, num_action))
end

function dynamics!(d::Dynamics, state, action, parameter)
    d.evaluate(d.evaluate_cache, state, action, parameter)
    return d.evaluate_cache
end

function jacobian!(jacobian_states, jacobian_actions, dynamics::Vector{Dynamics{T}}, states, actions, parameters) where T
    for (t, d) in enumerate(dynamics)
        d.jacobian_state(d.jacobian_state_cache, states[t], actions[t], parameters[t])
        d.jacobian_action(d.jacobian_action_cache, states[t], actions[t], parameters[t])
        @views jacobian_states[t] .= d.jacobian_state_cache
        @views jacobian_actions[t] .= d.jacobian_action_cache
        fill!(d.jacobian_state_cache, 0.0) # TODO: confirm this is necessary
        fill!(d.jacobian_action_cache, 0.0) # TODO: confirm this is necessary
    end
end

num_trajectory(dynamics::Vector{Dynamics{T}}) where T = sum([d.num_state + d.num_action for d in dynamics]) + dynamics[end].num_next_state

# user-provided dynamics and gradients
function Dynamics(f::Function, fx::Function, fu::Function, num_next_state::Int, num_state::Int, num_action::Int,
    num_parameter::Int=0)
    return Dynamics(f, fx, fu,
                    num_next_state, num_state, num_action, num_parameter,
                    zeros(num_next_state), zeros(num_next_state, num_state), zeros(num_next_state, num_action))
end
