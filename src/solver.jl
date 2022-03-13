"""
    Problem Data
"""
mutable struct Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
    problem::ProblemData{T,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
	policy::PolicyData{N,M,NN,MM,MN,NNN,MNN}
	data::SolverData{T}
    options::Options{T}
end

function solver(model::Model{T}, obj::Objective{T}; 
    w=[[zeros(d.nw) for d in model]..., zeros(0)],
    opts=Options{T}()) where T

	# allocate policy data
    policy = policy_data(model)

    # allocate problem data
    problem = problem_data(model, obj, w=w)

    # allocate solver data
    data = solver_data(model)

	Solver(problem, policy, data, options)
end


function get_trajectory(solver::Solver)
	return solver.problem.x̄, solver.problem.ū[1:end-1]
end

function current_trajectory(solver::Solver)
	return solver.problem.x, solver.problem.u[1:end-1]
end

function solver(model::Model{T}, obj::Objective{T}, cons::Constraints{T};
    w=[[zeros(d.nw) for d in model]..., zeros(0)],
    opts=Options{T}()) where T

	# augmented Lagrangian
	obj_al = augmented_lagrangian(model, obj, cons)

    # allocate policy data  
    policy = policy_data(model)

    # allocate model data
    problem = problem_data(model, obj_al, w=w)

    # allocate solver data
    data = solver_data(model)

	Solver(problem, policy, data, opts)
end

function initialize_controls!(solver::Solver, u) 
    for (t, ut) in enumerate(u) 
        solver.problem.ū[t] .= ut
    end 
end

function initialize_states!(solver::Solver, x) 
    for (t, xt) in enumerate(x)
        solver.problem.x̄[t] .= xt
    end
end