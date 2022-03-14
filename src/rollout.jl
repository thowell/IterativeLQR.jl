function rollout!(policy::PolicyData, problem::ProblemData; step_size=1.0)
    # model 
    model = problem.model.dynamics

    # trajectories
    x = problem.x
    u = problem.u
    w = problem.w
    x̄ = problem.x̄
    ū = problem.ū

    # policy
    K = policy.K
    k = policy.k

    # initial state
    x[1] .= x̄[1]

    # rollout
    for (t, dyn) in enumerate(model)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        u[t] .= k[t] 
        u[t] .*= step_size 
        u[t] .+= ū[t] 
        mul!(u[t], K[t], x[t], 1.0, 1.0) 
        mul!(u[t], K[t], x̄[t], -1.0, 1.0)
        x[t+1] .= dynamics!(dyn, x[t], u[t], w[t])
    end
end

function rollout(model::Model, x1, u, w=[zeros(d.nw) for d in model])
    x_hist = [x1]
    for (t, dyn) in enumerate(model) 
        push!(x_hist, copy(dynamics!(dyn, x_hist[end], u[t], w[t])))
    end
    return x_hist
end
