function cost(data::ProblemData; 
    mode=:nominal)

    if mode == :nominal
        return cost(data.objective.costs, data.x̄, data.ū, data.w)
    elseif mode == :current
        return cost(data.objective.costs, data.x, data.u, data.w)
    else 
        return 0.0 
    end
end

function cost!(solver_data::SolverData, problem::ProblemData; 
    mode=:nominal)

	if mode == :nominal
		solver_data.obj[1] = cost(problem.objective.costs, problem.x̄, problem.ū, problem.w)
	elseif mode == :current
		solver_data.obj[1] = cost(problem.objective.costs, problem.x, problem.u, problem.w)
	end

	if problem.objective.costs isa AugmentedLagrangianCosts
		solver_data.c_max[1] = constraint_violation(
            problem.objective.costs.constraint_data,
			problem.x, problem.u, problem.w,
			norm_type = Inf)
	end

	return solver_data.obj
end

function update_nominal_trajectory!(data::ProblemData) 
    T = length(data.x) 
    for t = 1:T 
        data.x̄[t] .= data.x[t] 
        t == T && continue 
        data.ū[t] .= data.u[t] 
    end 
end

#TODO: clean up
function trajectory_sensitivities(problem::ProblemData, policy::PolicyData, solver_data::SolverData)
    T = length(problem.x)
    fill!(problem.z, 0.0)
    for t = 1:T-1
        zx = @views problem.z[solver_data.idx_x[t]]
        zu = @views problem.z[solver_data.idx_u[t]]
        zy = @views problem.z[solver_data.idx_x[t+1]]
        zu .= policy.k[t] 
        mul!(zu, policy.K[t], zx, 1.0, 1.0)
        mul!(zy, problem.model.jacobian_action[t], zu)
        mul!(zy, problem.model.jacobian_state[t], zx, 1.0, 1.0)
        # problem.z[solver_data.idx_u[t]] .= policy.K[t] * problem.z[solver_data.idx_x[t]] + policy.k[t]
        # problem.z[solver_data.idx_x[t+1]] .= problem.model.jacobian_state[t] * problem.z[solver_data.idx_x[t]] + problem.model.jacobian_action[t] * problem.z[solver_data.idx_u[t]]
    end
end

function trajectories(problem::ProblemData; 
    mode=:nominal) 
    x = mode == :nominal ? problem.x̄ : problem.x 
    u = mode == :nominal ? problem.ū : problem.u 
    w = problem.w
    return x, u, w 
end