function ilqr_solve!(solver::Solver)

    # printstyled("Iterative LQR\n",
	# 	color=:red, bold=true)

	# data
	policy = solver.policy   
    problem = solver.problem
    reset!(problem.model)
    reset!(problem.objective) 
	data = solver.data
    solver.options.reset_cache && reset!(data)

	cost!(data, problem, 
        mode=:nominal)
    gradients!(problem, 
        mode=:nominal)
    backward_pass!(policy, problem, 
        mode=:nominal)

    obj_prev = data.obj[1]
    for i = 1:solver.options.max_iterations
        forward_pass!(policy, problem, data,
            min_step_size=solver.options.min_step_size,
            line_search=solver.options.line_search,
            verbose=solver.options.verbose)
        if solver.options.line_search != :none
            gradients!(problem, 
                mode=:nominal)
            backward_pass!(policy, problem, 
                mode=:nominal)
            lagrangian_gradient!(data, policy, problem)
        end

        # gradient norm
        gradient_norm = norm(data.gradient, Inf)

        # info
        data.iter[1] += 1
        solver.options.verbose && println(
            "iter:          $i
             cost:          $(data.obj[1])
			 gradient_norm: $(gradient_norm)
			 c_max:         $(data.c_max[1])
			 step_size:             $(data.step_size[1])")

        # check convergence
		gradient_norm < solver.options.lagrangian_gradient_tolerance && break
        abs(data.obj[1] - obj_prev) < solver.options.objective_tolerance ? break : (obj_prev = data.obj[1])
        !data.status[1] && break
    end

    return nothing
end

function ilqr_solve!(solver::Solver, x, u; kwargs...)
    initialize_controls!(solver, u) 
    initialize_states!(solver, x) 
    ilqr_solve!(solver; kwargs...)
end


"""
    gradient of Lagrangian
        https://web.stanford.edu/class/ee363/lectures/lqr-lagrange.pdf
"""
function lagrangian_gradient!(data::SolverData, policy::PolicyData, problem::ProblemData)
	p = policy.value.gradient
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    T = length(problem.x)

    for t = 1:T-1
        Lx = @views data.gradient[data.idx_x[t]]
        Lx .= Qx[t] 
        Lx .-= p[t] 
        Lu = @views data.gradient[data.idx_u[t]]
        Lu .= Qu[t]
        # data.gradient[data.idx_x[t]] = Qx[t] - p[t] # should always be zero by construction
        # data.gradient[data.idx_u[t]] = Qu[t]
    end
    # NOTE: gradient wrt xT is satisfied implicitly
end

"""
    augmented Lagrangian solve
"""
function constrained_ilqr_solve!(solver::Solver)

	# verbose && printstyled("Iterative LQR\n",
	# 	color=:red, bold=true)

    # reset solver cache 
    reset!(solver.data) 

    # reset duals 
    for (t, constraint_dual) in enumerate(solver.problem.objective.costs.constraint_dual)
        fill!(constraint_dual, 0.0)
	end

	# initialize penalty
	for (t, constraint_penalty) in enumerate(solver.problem.objective.costs.constraint_penalty)
        fill!(constraint_penalty, solver.options.initial_constraint_penalty)
	end

	for i = 1:solver.options.max_dual_updates
		solver.options.verbose && println("  al iter: $i")

		# primal minimization
		ilqr_solve!(solver)

		# update trajectories
		cost!(solver.data, solver.problem, 
            mode=:nominal)
		
        # constraint violation
		solver.data.c_max[1] <= solver.options.constraint_tolerance && break

        # dual ascent
		augmented_lagrangian_update!(solver.problem.objective.costs,
			scaling_penalty=solver.options.scaling_penalty, 
            max_penalty=solver.options.max_penalty)
	end

    return nothing
end

function constrained_ilqr_solve!(solver::Solver, x, u; kwargs...)
    initialize_controls!(solver, u) 
    initialize_states!(solver, x) 
    constrained_ilqr_solve!(solver; kwargs...)
end

function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:Objective{T}}
    iterative_lqr!(solver, args...; kwargs...)
end

function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:AugmentedLagrangianCosts{T}}
    constrained_ilqr_solve!(solver, args...; kwargs...)
end



