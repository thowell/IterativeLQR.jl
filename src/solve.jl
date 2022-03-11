function ilqr_solve!(solver::Solver)

    # printstyled("Iterative LQR\n",
	# 	color=:red, bold=true)

	# data
	p_data = solver.p_data   
    m_data = solver.m_data
    reset!(m_data.model_deriv)
    reset!(m_data.obj_deriv) 
	s_data = solver.s_data
    solver.options.reset_cache && reset!(s_data)

	objective!(s_data, m_data, mode=:nominal)
    derivatives!(m_data, mode=:nominal)
    backward_pass!(p_data, m_data, mode=:nominal)

    obj_prev = s_data.obj[1]
    for i = 1:solver.options.max_iter
        forward_pass!(p_data, m_data, s_data,
            α_min=solver.options.α_min,
            linesearch=solver.options.linesearch,
            verbose=solver.options.verbose)
        if solver.options.linesearch != :none
            derivatives!(m_data, mode=:nominal)
            backward_pass!(p_data, m_data, mode=:nominal)
            lagrangian_gradient!(s_data, p_data, m_data)
        end

        # gradient norm
        grad_norm = norm(s_data.gradient, Inf)

        # info
        s_data.iter[1] += 1
        solver.options.verbose && println("     iter: $i
             cost: $(s_data.obj[1])
			 grad_norm: $(grad_norm)
			 c_max: $(s_data.c_max[1])
			 α: $(s_data.α[1])")

        # check convergence
		grad_norm < solver.options.grad_tol && break
        abs(s_data.obj[1] - obj_prev) < solver.options.obj_tol ? break : (obj_prev = s_data.obj[1])
        !s_data.status[1] && break
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
function lagrangian_gradient!(s_data::SolverData, p_data::PolicyData, m_data::ModelData)
	p = p_data.p
    Qx = p_data.Qx
    Qu = p_data.Qu
    T = length(m_data.x)

    for t = 1:T-1
        Lx = @views s_data.gradient[s_data.idx_x[t]]
        Lx .= Qx[t] 
        Lx .-= p[t] 
        Lu = @views s_data.gradient[s_data.idx_u[t]]
        Lu .= Qu[t]
        # s_data.gradient[s_data.idx_x[t]] = Qx[t] - p[t] # should always be zero by construction
        # s_data.gradient[s_data.idx_u[t]] = Qu[t]
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
    reset!(solver.s_data) 

    # reset duals 
    for (t, λ) in enumerate(solver.m_data.obj.λ)
        fill!(λ, 0.0)
	end

	# initialize penalty
	for (t, ρ) in enumerate(solver.m_data.obj.ρ)
        fill!(ρ, solver.options.ρ_init)
	end

	for i = 1:solver.options.max_al_iter
		solver.options.verbose && println("  al iter: $i")

		# primal minimization
		ilqr_solve!(solver)

		# update trajectories
		objective!(solver.s_data, solver.m_data, mode=:nominal)
		
        # constraint violation
		solver.s_data.c_max[1] <= solver.options.con_tol && break

        # dual ascent
		augmented_lagrangian_update!(solver.m_data.obj,
			s=solver.options.ρ_scale, max_penalty=solver.options.ρ_max)
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



