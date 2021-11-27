function forward_pass!(p_data::PolicyData, m_data::ModelData, s_data::SolverData;
    linesearch = :armijo,
    α_min = 1.0e-5,
    c1 = 1.0e-4,
    c2 = 0.9,
    max_iter = 25)

    # reset solver status
    s_data.status[1] = false

    # previous cost
    J_prev = s_data.obj[1]

    # gradient of Lagrangian
    lagrangian_gradient!(s_data, p_data, m_data)

    if linesearch == :armijo
        Δz!(m_data, p_data, s_data)
        delta_grad_product = s_data.gradient' * m_data.z
    else
        delta_grad_product = 0.0
    end

    # line search with rollout
    s_data.α[1] = 1.0
    iter = 1
    while s_data.α[1] >= α_min
        iter > max_iter && (@warn "forward pass failure", break)

        J = Inf
        #TODO: remove try-catch
        try
            rollout!(p_data, m_data, α=s_data.α[1])
            J = objective!(s_data, m_data, mode=:current)[1]
        catch
            @warn "rollout failure"
            @show norm(s_data.gradient)
        end
        if (J <= J_prev + c1 * s_data.α[1] * delta_grad_product)
            # update nominal
            update_nominal_trajectory!(m_data)
            s_data.obj[1] = J
            s_data.status[1] = true
            break
        else
            s_data.α[1] *= 0.5
            iter += 1
        end
    end
    s_data.α[1] < α_min && (@warn "line search failure")
end

