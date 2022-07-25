struct Constraint{T}
    evaluate
    jacobian_state
    jacobian_action
    num_constraint::Int
    num_state::Int
    num_action::Int
    num_parameter::Int
    evaluate_cache::Vector{T}
    jacobian_state_cache::Matrix{T}
    jacobian_action_cache::Matrix{T}
    indices_inequality::Vector{Int}
end

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(f::Function, num_state::Int, num_action::Int;
    indices_inequality::Vector{Int}=collect(1:0),
    num_parameter::Int=0)

    #TODO: option to load/save methods
    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_action)
    w = Symbolics.variables(:w, 1:num_parameter)
    # @variables x[1:num_state], u[1:num_action], w[1:num_parameter]

    evaluate = num_parameter > 0 ? f(x, u, w) : f(x, u)
    jacobian_state = Symbolics.jacobian(evaluate, x)
    jacobian_action = Symbolics.jacobian(evaluate, u)

    evaluate_func = eval(Symbolics.build_function(evaluate, x, u, w)[2])
    jacobian_state_func = eval(Symbolics.build_function(jacobian_state, x, u, w)[2])
    jacobian_action_func = eval(Symbolics.build_function(jacobian_action, x, u, w)[2])

    num_constraint = length(evaluate)

    return Constraint(
        evaluate_func,
        jacobian_state_func, jacobian_action_func,
        num_constraint, num_state, num_action, num_parameter,
        zeros(num_constraint), zeros(num_constraint, num_state), zeros(num_constraint, num_action),
        indices_inequality)
end

function Constraint()
    return Constraint(
        (c, x, u, w) -> nothing,
        (jx, x, u, w) -> nothing, (ju, x, u, w) -> nothing,
        0, 0, 0, 0,
        Float64[], Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0),
        collect(1:0))
end

function Constraint(f::Function, fx::Function, fu::Function, num_constraint::Int, num_state::Int, num_action::Int;
    indices_inequality::Vector{Int}=collect(1:0),
    num_parameter::Int=0)

    return Constraint(
        f,
        fx, fu,
        num_constraint, num_state, num_action, num_parameter,
        zeros(num_constraint), zeros(num_constraint, num_state), zeros(num_constraint, num_action),
        indices_inequality)
end

function constraint!(violations, constraints::Constraints{T}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        con.num_constraint == 0 && continue
        con.evaluate(con.evaluate_cache, states[t], actions[t], parameters[t])
        @views violations[t] .= con.evaluate_cache
        fill!(con.evaluate_cache, 0.0) # TODO: confirm this is necessary
    end
end

function jacobian!(jacobian_states, jacobian_actions, constraints::Constraints{T}, states, actions, parameters) where T
    H = length(constraints)
    for (t, con) in enumerate(constraints)
        con.num_constraint == 0 && continue
        con.jacobian_state(con.jacobian_state_cache, states[t], actions[t], parameters[t])
        @views jacobian_states[t] .= con.jacobian_state_cache
        fill!(con.jacobian_state_cache, 0.0) # TODO: confirm this is necessary
        t == H && continue
        con.jacobian_action(con.jacobian_action_cache, states[t], actions[t], parameters[t])
        @views jacobian_actions[t] .= con.jacobian_action_cache
        fill!(con.jacobian_action_cache, 0.0) # TODO: confirm this is necessary
    end
end
