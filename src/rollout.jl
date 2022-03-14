function rollout!(policy::PolicyData, problem::ProblemData; 
    step_size=1.0)

    # model 
    dynamics = problem.model.dynamics

    # trajectories
    x = problem.states
    u = problem.actions
    w = problem.parameters
    x̄ = problem.nominal_states
    ū = problem.nominal_actions

    # policy
    K = policy.K
    k = policy.k

    # initial state
    x[1] .= x̄[1]

    # rollout
    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        u[t] .= k[t] 
        u[t] .*= step_size 
        u[t] .+= ū[t] 
        mul!(u[t], K[t], x[t], 1.0, 1.0) 
        mul!(u[t], K[t], x̄[t], -1.0, 1.0)
        x[t+1] .= dynamics!(d, x[t], u[t], w[t])
    end
end

function rollout(dynamics::Vector{Dynamics{T}}, initial_state, actions, 
    parameters=[zeros(d.nw) for d in dynamics]) where T

    x_history = [initial_state]
    for (t, d) in enumerate(dynamics) 
        push!(x_history, copy(dynamics!(d, x_history[end], actions[t], parameters[t])))
    end
    
    return x_history
end
