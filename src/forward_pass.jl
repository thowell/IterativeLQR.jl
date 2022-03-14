function forward_pass!(policy::PolicyData, problem::ProblemData, solver_data::SolverData;
    line_search=:armijo,
    min_step_size=1.0e-5,
    c1=1.0e-4,
    c2=0.9,
    max_iterations=25,
    verbose=false)

    # reset solver status
    solver_data.status[1] = false

    # previous cost
    J_prev = solver_data.obj[1]

    # gradient of Lagrangian
    lagrangian_gradient!(solver_data, policy, problem)

    if line_search == :armijo
        trajectory_sensitivities(problem, policy, solver_data)
        delta_grad_product = solver_data.gradient' * problem.z
    else
        delta_grad_product = 0.0
    end

    # line search with rollout
    solver_data.step_size[1] = 1.0
    iter = 1
    while solver_data.step_size[1] >= min_step_size
        iter > max_iterations && (verbose && (@warn "forward pass failure"), break)

        J = Inf
        #TODO: remove try-catch
        try
            rollout!(policy, problem, step_size=solver_data.step_size[1])
            J = cost!(solver_data, problem, mode=:current)[1]
        catch
            if verbose
                @warn "rollout failure"
                @show norm(solver_data.gradient)
            end
        end
        if (J <= J_prev + c1 * solver_data.step_size[1] * delta_grad_product)
            # update nominal
            update_nominal_trajectory!(problem)
            solver_data.obj[1] = J
            solver_data.status[1] = true
            break
        else
            solver_data.step_size[1] *= 0.5
            iter += 1
        end
    end
    solver_data.step_size[1] < min_step_size && (verbose && (@warn "line search failure"))
end

