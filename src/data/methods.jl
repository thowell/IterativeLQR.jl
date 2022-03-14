function cost(problem::ProblemData; 
    mode=:nominal)

    if mode == :nominal
        return cost(problem.objective.costs, problem.nominal_states, problem.nominal_actions, problem.parameters)
    elseif mode == :current
        return cost(problem.objective.costs, problem.states, problem.actions, problem.parameters)
    else 
        return 0.0 
    end
end

function cost!(data::SolverData, problem::ProblemData; 
    mode=:nominal)

	if mode == :nominal
		data.objective[1] = cost(problem.objective.costs, problem.nominal_states, problem.nominal_actions, problem.parameters)
	elseif mode == :current
		data.objective[1] = cost(problem.objective.costs, problem.states, problem.actions, problem.parameters)
	end

	if problem.objective.costs isa AugmentedLagrangianCosts
		data.max_violation[1] = constraint_violation(
            problem.objective.costs.constraint_data,
			problem.states, problem.actions, problem.parameters,
			norm_type = Inf)
	end

	return data.objective
end

function update_nominal_trajectory!(data::ProblemData) 
    H = length(data.states) 
    for t = 1:H 
        data.nominal_states[t] .= data.states[t] 
        t == H && continue 
        data.nominal_actions[t] .= data.actions[t] 
    end 
end

#TODO: clean up
function trajectory_sensitivities(problem::ProblemData, policy::PolicyData, data::SolverData)
    H = length(problem.states)
    fill!(problem.trajectory, 0.0)
    for t = 1:H-1
        zx = @views problem.trajectory[data.indices_state[t]]
        zu = @views problem.trajectory[data.indices_action[t]]
        zy = @views problem.trajectory[data.indices_state[t+1]]
        zu .= policy.k[t] 
        mul!(zu, policy.K[t], zx, 1.0, 1.0)
        mul!(zy, problem.model.jacobian_action[t], zu)
        mul!(zy, problem.model.jacobian_state[t], zx, 1.0, 1.0)
    end
end

function trajectories(problem::ProblemData; 
    mode=:nominal) 
    x = mode == :nominal ? problem.nominal_states : problem.states
    u = mode == :nominal ? problem.nominal_actions : problem.actions
    w = problem.parameters
    return x, u, w 
end