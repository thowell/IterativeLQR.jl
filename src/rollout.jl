function rollout!(p_data::PolicyData, m_data::ModelData; α=1.0)
    # model 
    model = m_data.model

    # trajectories
    x = m_data.x
    u = m_data.u
    w = m_data.w
    x̄ = m_data.x̄
    ū = m_data.ū

    # policy
    K = p_data.K
    k = p_data.k

    # initial state
    x[1] .= x̄[1]

    # rollout
    for (t, dyn) in enumerate(model)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + α * k[t]
        u[t] .= k[t] 
        u[t] .*= α 
        u[t] .+= ū[t] 
        mul!(u[t], K[t], x[t], 1.0, 1.0) 
        mul!(u[t], K[t], x̄[t], -1.0, 1.0)
        x[t+1] .= step!(dyn, x[t], u[t], w[t])
    end
end

function rollout(model::Model, x1, u, w)
    x_hist = [x1]
    for (t, dyn) in enumerate(model) 
        push!(x_hist, copy(step!(dyn, x_hist[end], u[t], w[t])))
    end
    return x_hist
end
