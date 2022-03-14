function gradients!(dynamics::Model, data::ProblemData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    jx = data.model.jacobian_state
    ju = data.model.jacobian_action
    jacobian!(jx, ju, dynamics, x, u, w)
end

function gradients!(obj::Objective, data::ProblemData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    gradx = data.objective.gradient_state
    gradu = data.objective.gradient_action
    hessxx = data.objective.hessian_state_state
    hessuu = data.objective.hessian_action_action
    hessux = data.objective.hessian_action_state
    cost_gradient!(gradx, gradu, obj, x, u, w)
    cost_hessian!(hessxx, hessuu, hessux, obj, x, u, w) 
end

function gradients!(obj::AugmentedLagrangianCosts, data::ProblemData; mode=:nominal)
    # objective 
    gradient_state = data.objective.gradient_state
    gradient_action = data.objective.gradient_action
    hessian_state_state = data.objective.hessian_state_state
    hessian_action_action = data.objective.hessian_action_action
    hessian_action_state = data.objective.hessian_action_state

    # constraints
    constraints = obj.constraint_data.constraints
    violations = obj.constraint_data.violations
    jacobian_state = obj.constraint_data.jacobian_state
    jacobian_action = obj.constraint_data.jacobian_action
    constraint_penalty = obj.constraint_penalty
    constraint_dual = obj.constraint_dual
    active_set = obj.active_set
    constraint_penalty_matrix = obj.constraint_penalty_matrix
    constraint_tmp = obj.constraint_tmp 
    constraint_jacobian_state_tmp = obj.constraint_jacobian_state_tmp 
    constraint_jacobian_action_tmp = obj.constraint_jacobian_action_tmp

    T = length(obj)

    # derivatives
    gradients!(obj.costs, data, 
        mode=mode)
    gradients!(obj.constraint_data, data, 
        mode=mode)

    for t = 1:T
        nc = constraints[t].nc
        for i = 1:nc 
            constraint_penalty_matrix[t][i, i] = constraint_penalty[t][i] * active_set[t][i]
        end
        constraint_tmp[t] .= constraint_dual[t] 

        # gradient_state
        mul!(constraint_tmp[t], constraint_penalty_matrix[t], violations[t], 1.0, 1.0)
        mul!(gradient_state[t], transpose(jacobian_state[t]), constraint_tmp[t], 1.0, 1.0)

        # hessian_state_state 
        mul!(constraint_jacobian_state_tmp[t], constraint_penalty_matrix[t], jacobian_state[t])
        mul!(hessian_state_state[t], transpose(jacobian_state[t]), constraint_jacobian_state_tmp[t], 1.0, 1.0)

        t == T && continue 

        # gradient_action 
        mul!(gradient_action[t], transpose(jacobian_action[t]), constraint_tmp[t], 1.0, 1.0) 

        # hessian_action_action 
        mul!(constraint_jacobian_action_tmp[t], constraint_penalty_matrix[t], jacobian_action[t]) 
        mul!(hessian_action_action[t], transpose(jacobian_action[t]), constraint_jacobian_action_tmp[t], 1.0, 1.0) 

        # hessian_action_state 
        mul!(hessian_action_state[t], transpose(jacobian_action[t]), constraint_jacobian_state_tmp[t], 1.0, 1.0)
    end
end

function gradients!(constraint_data::ConstraintsData, problem::ProblemData;
    mode=:nominal)
    x, u, w = trajectories(problem, 
        mode=mode)
    cx = constraint_data.jacobian_state
    cu = constraint_data.jacobian_action
    jacobian!(cx, cu, constraint_data.constraints, x, u, w)
end

function gradients!(problem::ProblemData; 
    mode=:nominal)
    gradients!(problem.model.dynamics, problem, 
        mode=mode)
    gradients!(problem.objective.costs, problem, 
        mode=mode)
end
