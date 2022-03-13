function objective(data::ProblemData; 
    mode = :nominal)
    if mode == :nominal
        return objective(data.objective.costs, data.x̄, data.ū, data.w)
    elseif mode == :current
        return objective(data.objective.costs, data.x, data.u, data.w)
    else 
        return 0.0 
    end
end

function objective!(solver_data::SolverData, problem_data::ProblemData; 
    mode=:nominal)
	if mode == :nominal
		solver_data.obj[1] = objective(problem_data.objective.costs, problem_data.x̄, problem_data.ū, problem_data.w)
	elseif mode == :current
		solver_data.obj[1] = objective(problem_data.objective.costs, problem_data.x, problem_data.u, problem_data.w)
	end

	if problem_data.objective.costs isa AugmentedLagrangianCosts
		solver_data.c_max[1] = constraint_violation(
            problem_data.objective.costs.constraint_data,
			problem_data.x, problem_data.u, problem_data.w,
			norm_type = Inf)
	end

	return solver_data.obj
end

function trajectories(problem_data::ProblemData; 
    mode=:nominal) 
    x = mode == :nominal ? problem_data.x̄ : problem_data.x 
    u = mode == :nominal ? problem_data.ū : problem_data.u 
    w = problem_data.w
    return x, u, w 
end

#TODO: clean up
function trajectory_sensitivities(problem_data::ProblemData, policy_data::PolicyData, solver_data::SolverData)
    T = length(problem_data.x)
    fill!(problem_data.z, 0.0)
    for t = 1:T-1
        zx = @views problem_data.z[solver_data.idx_x[t]]
        zu = @views problem_data.z[solver_data.idx_u[t]]
        zy = @views problem_data.z[solver_data.idx_x[t+1]]
        zu .= policy_data.k[t] 
        mul!(zu, policy_data.K[t], zx, 1.0, 1.0)
        mul!(zy, problem_data.model.fu[t], zu)
        mul!(zy, problem_data.model.fx[t], zx, 1.0, 1.0)
        # problem_data.z[solver_data.idx_u[t]] .= policy_data.K[t] * problem_data.z[solver_data.idx_x[t]] + policy_data.k[t]
        # problem_data.z[solver_data.idx_x[t+1]] .= problem_data.model.fx[t] * problem_data.z[solver_data.idx_x[t]] + problem_data.model.fu[t] * problem_data.z[solver_data.idx_u[t]]
    end
end

function update_nominal_trajectory!(data::ProblemData) 
    T = length(data.x) 
    for t = 1:T 
        data.x̄[t] .= data.x[t] 
        t == T && continue 
        data.ū[t] .= data.u[t] 
    end 
end