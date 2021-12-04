function ilqr_solve!(prob::ProblemData;
    max_iter=10,
    obj_tol=1.0e-3,
    grad_tol=1.0e-3,
    α_min=1.0e-5,
    linesearch=:armijo,
    reset_cache=true,
    verbose=false)

    # printstyled("Iterative LQR\n",
	# 	color=:red, bold=true)

	# data
	p_data = prob.p_data   
    m_data = prob.m_data
    reset!(m_data.model_deriv)
    reset!(m_data.obj_deriv) 
	s_data = prob.s_data
    reset_cache && reset!(s_data)

	objective!(s_data, m_data, mode=:nominal)
    derivatives!(m_data, mode=:nominal)
    backward_pass!(p_data, m_data, mode=:nominal)

    obj_prev = s_data.obj[1]
    for i = 1:max_iter
        forward_pass!(p_data, m_data, s_data,
            α_min=α_min,
            linesearch=linesearch,
            verbose=verbose)
        if linesearch != :none
            derivatives!(m_data, mode=:nominal)
            backward_pass!(p_data, m_data, mode=:nominal)
            lagrangian_gradient!(s_data, p_data, m_data)
        end

        # gradient norm
        grad_norm = norm(s_data.gradient, Inf)

        # info
        s_data.iter[1] += 1
        verbose && println("     iter: $i
             cost: $(s_data.obj[1])
			 grad_norm: $(grad_norm)
			 c_max: $(s_data.c_max[1])
			 α: $(s_data.α[1])")

        # check convergence
		grad_norm < grad_tol && break
        abs(s_data.obj[1] - obj_prev) < obj_tol ? break : (obj_prev = s_data.obj[1])
        !s_data.status[1] && break
    end

    return nothing
end

function ilqr_solve!(prob::ProblemData, x, u; kwargs...)
    initialize_controls!(prob, u) 
    initialize_states!(prob, x) 
    ilqr_solve!(prob; kwargs...)
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
function constrained_ilqr_solve!(prob::ProblemData;
    linesearch=:armijo,
    max_iter=10,
	max_al_iter=10,
    α_min=1.0e-5,
    obj_tol=1.0e-3,
    grad_tol=1.0e-3,
	con_tol=1.0e-3,
	con_norm_type=Inf,
	ρ_init=1.0,
	ρ_scale=10.0,
	ρ_max=1.0e8,
    verbose=false)

	# verbose && printstyled("Iterative LQR\n",
	# 	color=:red, bold=true)

    # reset solver cache 
    reset!(prob.s_data) 

    # reset duals 
    for (t, λ) in enumerate(prob.m_data.obj.λ)
        fill!(λ, 0.0)
	end

	# initialize penalty
	for (t, ρ) in enumerate(prob.m_data.obj.ρ)
        fill!(ρ, ρ_init)
	end

	for i = 1:max_al_iter
		verbose && println("  al iter: $i")

		# primal minimization
		ilqr_solve!(prob,
            linesearch=linesearch,
            α_min=α_min,
		    max_iter=max_iter,
            obj_tol=obj_tol,
		    grad_tol=grad_tol,
            reset_cache=false,
		    verbose=verbose)

		# update trajectories
		objective!(prob.s_data, prob.m_data, mode=:nominal)
		
        # constraint violation
		prob.s_data.c_max[1] <= con_tol && break

        # dual ascent
		augmented_lagrangian_update!(prob.m_data.obj,
			s=ρ_scale, max_penalty=ρ_max)
	end

    return nothing
end

function constrained_ilqr_solve!(prob::ProblemData, x, u; kwargs...)
    initialize_controls!(prob, u) 
    initialize_states!(prob, x) 
    constrained_ilqr_solve!(prob; kwargs...)
end

function solve!(prob::ProblemData{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:Objective{T}}
    iterative_lqr!(prob, args...; kwargs...)
end

function solve!(prob::ProblemData{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:AugmentedLagrangianCosts{T}}
    constrained_ilqr_solve!(prob, args...; kwargs...)
end



