function gradients!(dynamics::Model, data::ProblemData; 
        mode=:nominal)
    x, u, w = trajectories(data, 
        mode=mode)
    jx = data.model.jacobian_state
    ju = data.model.jacobian_action
    jacobian!(jx, ju, dynamics, x, u, w)
end

function gradients!(obj::Objective, data::ProblemData; 
        mode=:nominal)
    x, u, w = trajectories(data, 
        mode=mode)
    gx = data.objective.gradient_state
    gu = data.objective.gradient_action
    gxx = data.objective.hessian_state_state
    guu = data.objective.hessian_action_action
    gux = data.objective.hessian_action_state
    cost_gradient!(gx, gu, obj, x, u, w)
    cost_hessian!(gxx, guu, gux, obj, x, u, w) 
end

function gradients!(obj::AugmentedLagrangianCosts, data::ProblemData; 
    mode=:nominal)
    # objective 
    gx = data.objective.gradient_state
    gu = data.objective.gradient_action
    gxx = data.objective.hessian_state_state
    guu = data.objective.hessian_action_action
    gux = data.objective.hessian_action_state

    # constraints
    cons = obj.constraint_data.constraints
    c = obj.constraint_data.violations
    cx = obj.constraint_data.jacobian_state
    cu = obj.constraint_data.jacobian_action
    ρ = obj.constraint_penalty
    λ = obj.constraint_dual
    a = obj.active_set
    Iρ = obj.constraint_penalty_matrix
    c_tmp = obj.constraint_tmp 
    cx_tmp = obj.constraint_jacobian_state_tmp 
    cu_tmp = obj.constraint_jacobian_action_tmp

    # horizon
    H = length(obj)

    # derivatives
    gradients!(obj.costs, data, 
        mode=mode)
    gradients!(obj.constraint_data, data, 
        mode=mode)

    for t = 1:H
        nc = cons[t].nc
        for i = 1:nc 
            Iρ[t][i, i] = ρ[t][i] * a[t][i]
        end
        c_tmp[t] .= λ[t] 

        # gradient_state
        mul!(c_tmp[t], Iρ[t], c[t], 1.0, 1.0)
        mul!(gx[t], transpose(cx[t]), c_tmp[t], 1.0, 1.0)

        # hessian_state_state 
        mul!(cx_tmp[t], Iρ[t], cx[t])
        mul!(gxx[t], transpose(cx[t]), cx_tmp[t], 1.0, 1.0)

        t == H && continue 

        # gradient_action 
        mul!(gu[t], transpose(cu[t]), c_tmp[t], 1.0, 1.0) 

        # hessian_action_action 
        mul!(cu_tmp[t], Iρ[t], cu[t]) 
        mul!(guu[t], transpose(cu[t]), cu_tmp[t], 1.0, 1.0) 

        # hessian_action_state 
        mul!(gux[t], transpose(cu[t]), cx_tmp[t], 1.0, 1.0)
    end
end

function gradients!(constraints_data::ConstraintsData, problem::ProblemData;
    mode=:nominal)
    x, u, w = trajectories(problem, 
        mode=mode)
    cx = constraints_data.jacobian_state
    cu = constraints_data.jacobian_action
    jacobian!(cx, cu, constraints_data.constraints, x, u, w)
end

function gradients!(problem::ProblemData; 
    mode=:nominal)
    gradients!(problem.model.dynamics, problem, 
        mode=mode)
    gradients!(problem.objective.costs, problem, 
        mode=mode)
end
