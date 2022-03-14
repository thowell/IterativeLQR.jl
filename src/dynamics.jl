struct Dynamics{T}
    val 
    jacobian_state 
    jacobian_action
    ny::Int 
    nx::Int 
    nu::Int
    nw::Int
    val_cache::Vector{T} 
    jacobian_state_cache::Matrix{T}
    jacobian_action_cache::Matrix{T}
end

Model{T} = Vector{Dynamics{T}} where T

function Dynamics(f::Function, nx::Int, nu::Int; nw::Int=0)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu], w[1:nw] 
    y = f(x, u, w) 
    jacobian_state = Symbolics.jacobian(y, x);
    jacobian_action = Symbolics.jacobian(y, u);
    val_func = eval(Symbolics.build_function(y, x, u, w)[2]);
    jacobian_state_func = eval(Symbolics.build_function(jacobian_state, x, u, w)[2]);
    jacobian_action_func = eval(Symbolics.build_function(jacobian_action, x, u, w)[2]);
    ny = length(y)

    return Dynamics(val_func, jacobian_state_func, jacobian_action_func, 
                    ny, nx, nu, nw, 
                    zeros(ny), zeros(ny, nx), zeros(ny, nu))
end

function dynamics!(d::Dynamics, state, action, parameter) 
    d.val(d.val_cache, state, action, parameter)
    return d.val_cache
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

num_var(dynamics::Vector{Dynamics{T}}) where T = sum([d.nx + d.nu for d in dynamics]) + dynamics[end].ny

# user-provided dynamics and gradients
function Dynamics(f::Function, fx::Function, fu::Function, ny::Int, nx::Int, nu::Int, 
    nw::Int=0)  
    return Dynamics(f, fx, fu, 
                    ny, nx, nu, nw, 
                    zeros(ny), zeros(ny, nx), zeros(ny, nu))
end