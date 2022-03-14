"""
    Problem Data
"""
mutable struct Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
    problem::ProblemData{T,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
	policy::PolicyData{N,M,NN,MM,MN,NNN,MNN}
	data::SolverData{T}
    options::Options{T}
end

function solver(dynamics::Vector{Dynamics{T}}, obj::Objective{T}; 
    parameters=[[zeros(d.num_parameter) for d in dynamics]..., zeros(0)],
    options=Options{T}()) where T

	# allocate policy data
    policy = policy_data(dynamics)

    # allocate problem data
    problem = problem_data(dynamics, obj, 
        parameters=parameters)

    # allocate solver data
    data = solver_data(dynamics)

	Solver(problem, policy, data, options)
end


function get_trajectory(solver::Solver)
	return solver.problem.nominal_states, solver.problem.nominal_actions[1:end-1]
end

function current_trajectory(solver::Solver)
	return solver.problem.states, solver.problem.actions[1:end-1]
end

function solver(dynamics::Vector{Dynamics{T}}, costs::Vector{Cost{T}}, constraints::Constraints{T};
    parameters=[[zeros(d.num_parameter) for d in dynamics]..., zeros(0)],
    options=Options{T}()) where T

	# augmented Lagrangian
	objective_al = augmented_lagrangian(dynamics, costs, constraints)

    # allocate policy data  
    policy = policy_data(dynamics)

    # allocate model data
    problem = problem_data(dynamics, objective_al, 
        parameters=parameters)

    # allocate solver data
    data = solver_data(dynamics)

	Solver(problem, policy, data, options)
end

function initialize_controls!(solver::Solver, actions) 
    for (t, ut) in enumerate(actions) 
        solver.problem.nominal_actions[t] .= ut
    end 
end

function initialize_states!(solver::Solver, states) 
    for (t, xt) in enumerate(states)
        solver.problem.nominal_states[t] .= xt
    end
end