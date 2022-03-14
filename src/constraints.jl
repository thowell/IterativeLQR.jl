struct Constraint{T}
    val 
    jacobian_state 
    jacobian_action
    nc::Int
    nx::Int 
    nu::Int 
    nw::Int
    val_cache::Vector{T} 
    jacobian_state_cache::Matrix{T}
    jacobian_action_cache::Matrix{T}
    indices_inequality::Vector{Int}
end

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(f::Function, nx::Int, nu::Int; 
    indices_inequality::Vector{Int}=collect(1:0), 
    nw::Int=0)

    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu], w[1:nw]
    
    val = f(x, u, w)
    jacobian_state = Symbolics.jacobian(val, x)
    jacobian_action = Symbolics.jacobian(val, u)

    val_func = eval(Symbolics.build_function(val, x, u, w)[2])
    jacobian_state_func = eval(Symbolics.build_function(jacobian_state, x, u, w)[2])
    jacobian_action_func = eval(Symbolics.build_function(jacobian_action, x, u, w)[2])

    nc = length(val) 
    
    return Constraint(
        val_func, 
        jacobian_state_func, jacobian_action_func,
        nc, nx, nu, nw,  
        zeros(nc), zeros(nc, nx), zeros(nc, nu), 
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

function constraints!(violations, constraints::Constraints{T}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        con.nc == 0 && continue
        con.val(con.val_cache, states[t], actions[t], parameters[t])
        @views violations[t] .= con.val_cache
        fill!(con.val_cache, 0.0) # TODO: confirm this is necessary 
    end
end

function jacobian!(jacobian_states, jacobian_actions, constraints::Constraints{T}, states, actions, parameters) where T
    H = length(constraints)
    for (t, con) in enumerate(constraints)
        con.nc == 0 && continue
        con.jacobian_state(con.jacobian_state_cache, states[t], actions[t], parameters[t])
        @views jacobian_states[t] .= con.jacobian_state_cache
        fill!(con.jacobian_state_cache, 0.0) # TODO: confirm this is necessary
        t == H && continue
        con.jacobian_action(con.jacobian_action_cache, states[t], actions[t], parameters[t])
        @views jacobian_actions[t] .= con.jacobian_action_cache
        fill!(con.jacobian_action_cache, 0.0) # TODO: confirm this is necessary
    end
end
