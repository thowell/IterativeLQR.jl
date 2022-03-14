function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData;
    line_search=:armijo,
    min_step_size=1.0e-5,
    c1=1.0e-4,
    c2=0.9,
    max_iterations=25,
    verbose=false)

    # reset solver status
    data.status[1] = false

    # previous cost
    J_prev = data.objective[1]

    # gradient of Lagrangian
    lagrangian_gradient!(data, policy, problem)

    if line_search == :armijo
        trajectory_sensitivities(problem, policy, data)
        delta_grad_product = data.gradient' * problem.trajectory
    else
        delta_grad_product = 0.0
    end

    # line search with rollout
    data.step_size[1] = 1.0
    iteration = 1
    while data.step_size[1] >= min_step_size
        iteration > max_iterations && (verbose && (@warn "forward pass failure"), break)

        J = Inf
        #TODO: remove try-catch
        # try
        rollout!(policy, problem, 
            step_size=data.step_size[1])
        J = cost!(data, problem, 
            mode=:current)[1]
        # catch
        #     if verbose
        #         @warn "rollout failure"
        #         @show norm(data.gradient)
        #     end
        # end
        if (J <= J_prev + c1 * data.step_size[1] * delta_grad_product)
            # update nominal
            update_nominal_trajectory!(problem)
            data.objective[1] = J
            data.status[1] = true
            break
        else
            data.step_size[1] *= 0.5
            iteration += 1
        end
    end
    data.step_size[1] < min_step_size && (verbose && (@warn "line search failure"))
end

