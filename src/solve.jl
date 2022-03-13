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
    for i = 1:solver.options.max_iter
        forward_pass!(policy, problem, data,
            α_min=solver.options.α_min,
            linesearch=solver.options.linesearch,
            verbose=solver.options.verbose)
        if solver.options.linesearch != :none
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
			 α:             $(data.α[1])")

        # check convergence
		gradient_norm < solver.options.grad_tol && break
        abs(data.obj[1] - obj_prev) < solver.options.obj_tol ? break : (obj_prev = data.obj[1])
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
	p = policy.p
    Qx = policy.Qx
    Qu = policy.Qu
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
    for (t, λ) in enumerate(solver.problem.objective.costs.λ)
        fill!(λ, 0.0)
	end

	# initialize penalty
	for (t, ρ) in enumerate(solver.problem.objective.costs.ρ)
        fill!(ρ, solver.options.ρ_init)
	end

	for i = 1:solver.options.max_al_iter
		solver.options.verbose && println("  al iter: $i")

		# primal minimization
		ilqr_solve!(solver)

		# update trajectories
		cost!(solver.data, solver.problem, 
            mode=:nominal)
		
        # constraint violation
		solver.data.c_max[1] <= solver.options.con_tol && break

        # dual ascent
		augmented_lagrangian_update!(solver.problem.objective.costs,
			s=solver.options.ρ_scale, 
            max_penalty=solver.options.ρ_max)
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



