"""
    Model Data
"""

struct ModelDerivativesData{X,U,W}
    fx::Vector{X}
    fu::Vector{U}
	fw::Vector{W}
end

function model_derivatives_data(model::Model)
	fx = [zeros(d.ny, d.nx) for d in model]
    fu = [zeros(d.ny, d.nu) for d in model]
	fw = [zeros(d.ny, d.nw) for d in model]
    ModelDerivativesData(fx, fu, fw)
end

struct ObjectiveDerivativesData{X,U,XX,UU,UX}
    gx::Vector{X}
    gu::Vector{U}
    gxx::Vector{XX}
    guu::Vector{UU}
    gux::Vector{UX}
end

function objective_derivatives_data(model::Model)
	gx = [[zeros(d.nx) for d in model]..., 
        zeros(model[end].ny)]
    gu = [zeros(d.nu) for d in model]
    gxx = [[zeros(d.nx, d.nx) for d in model]..., 
        zeros(model[end].ny, model[end].ny)]
    guu = [zeros(d.nu, d.nu) for d in model]
    gux = [zeros(d.nu, d.nx) for d in model]
    ObjectiveDerivativesData(gx, gu, gxx, guu, gux)
end

struct ModelData{T,X,U,D,FX,FU,FW,OX,OU,OXX,OUU,OUX}
    # current trajectory
    x::Vector{X}
    u::Vector{U}

    # disturbance trajectory
    w::Vector{D}

    # nominal trajectory
    x̄::Vector{X}
    ū::Vector{U}

    # dynamics model
    model::Model{T}

    # objective
    obj::Objective{T}

    # dynamics derivatives data
    model_deriv::ModelDerivativesData{FX,FU,FW}

    # objective derivatives data
    obj_deriv::ObjectiveDerivativesData{OX,OU,OXX,OUU,OUX}

    # z = (x1...,xT,u1,...,uT-1) | Δz = (Δx1...,ΔxT,Δu1,...,ΔuT-1)
    z::Vector{T}
end

function model_data(model::Model, obj::Objective; 
    w=[zeros(d.nw) for d in model])

	x = [[zeros(d.nx) for d in model]..., 
            zeros(model[end].ny)]
    u = [zeros(d.nu) for d in model]

    x̄ = [[zeros(d.nx) for d in model]..., 
            zeros(model[end].ny)]
    ū = [zeros(d.nu) for d in model]

    model_deriv = model_derivatives_data(model)
    obj_deriv = objective_derivatives_data(model)

    z = zeros(num_var(model))

    ModelData(x, u, w, x̄, ū, model, obj, dyn_deriv, obj_deriv, z)
end

function objective(data::ModelData; mode = :nominal)
    if mode == :nominal
        return eval_obj(data.obj, data.x̄, data.ū, data.w)
    elseif mode == :current
        return eval_obj(data.obj, data.x, data.u, data.w)
    else 
        return 0.0 
    end
end

"""
    Policy Data
"""
struct PolicyData{N,M,NN,MM,MN,NNN,MNN}
    # policy
    K::Vector{MN}
    k::Vector{M}

    K_cand::Vector{MN}
    k_cand::Vector{M}

    # value function approximation
    P::Vector{NN}
    p::Vector{N}

    # state-action value function approximation
    Qx::Vector{N}
    Qu::Vector{M}
    Qxx::Vector{NN}
    Quu::Vector{MM}
    Qux::Vector{MN}

	xx̂_tmp::Vector{NNN}
	ux̂_tmp::Vector{MNN}
	uu_tmp::Vector{MM}
	ux_tmp::Vector{MN}
end

function policy_data(model::Model)
	K = [zeros(d.nu, d.nx) for d in model]
    k = [zeros(d.nu) for d in model]

    K_cand = [zeros(d.nu, d.nx) for d in model]
    k_cand = [zeros(d.nu) for d in model]

    P = [[zeros(d.nx, d.nx) for d in model]..., 
            zeros(model[end].ny, model[end].ny)]
    p =  [[zeros(d.nx) for d in model]..., 
            zeros(model[end].ny)]

    Qx = [zeros(d.nx) for d in model]
    Qu = [zeros(d.nu) for d in model]
    Qxx = [zeros(d.nx, d.nx) for d in model]
    Quu = [zeros(d.nu, d.nu) for d in model]
    Qux = [zeros(d.nu, d.nx) for d in model]

	xx̂_tmp = [zeros(d.nx, d.ny) for d in model]
	ux̂_tmp = [zeros(d.nu, d.ny) for d in model]
	uu_tmp = [zeros(d.nu, d.nu) for d in model]
	ux_tmp = [zeros(d.nu, d.nx) for d in model]

    PolicyData(K, k, K_cand, k_cand, P, p, Qx, Qu, Qxx, Quu, Qux, xx̂_tmp, ux̂_tmp, uu_tmp, ux_tmp)
end

"""
    Solver Data
"""
struct SolverData{T}
    obj::Vector{T}              # objective value
    gradient::Vector{T}         # Lagrangian gradient
	c_max::Vector{T}            # maximum constraint violation

    idx_x::Vector{Vector{Int}}  # indices for state trajectory
    idx_u::Vector{Vector{Int}}  # indices for control trajectory

    α::Vector{T}                # step length
    status::Vector{Bool}        # solver status

	cache::Dict{Symbol,Vector{T}}       # solver stats
end

function solver_data(model::Model{T}; max_cache=1000) where T
    # indices x and u
    idx_x = Vector{Int}[]
    idx_u = Vector{Int}[] 
    n_sum = 0 
    m_sum = 0 
    n_total = sum([d.nx for d in model]) + model[end].ny
    for d in model
        push!(idx_x, collect(n_sum .+ (1:d.nx))) 
        push!(idx_u, collect(n_total + m_sum .+ (1:d.nu)))
        n_sum += d.nx 
        m_sum += d.nu 
    end
    push!(idx_x, collect(n_sum .+ (1:model[end].ny)))

    obj = [Inf]
	c_max = [0.0]
    α = [1.0]
    gradient = zeros(num_var(model))
	cache = Dict(:obj => zeros(max_cache), 
                 :gradient => zeros(max_cache), 
                 :c_max => zeros(max_cache), 
                 :α => zeros(max_cache),
                 :iter => 1)

    SolverData(obj, gradient, c_max, idx_x, idx_u, α, [false], cache)
end

function cache!(data::SolverData)
    iter = data.cache[:iter] 
    (iter > length(data[:obj])) && (@warn "solver data cache exceeded")
	data.cache[:obj][iter] = data.obj
	data.cache[:gradient][iter] = data.gradient
	data.cache[:c_max][iter] = data.c_max
	data.cache[:α][iter] = data.α
    data.cache[:iter] = iter + 1
    return nothing
end

function objective!(s_data::SolverData, m_data::ModelData; mode = :nominal)
	if mode == :nominal
		s_data.obj = eval_obj(m_data.obj, m_data.x̄, m_data.ū, m_data.w)
	elseif mode == :current
		s_data.obj = eval_obj(m_data.obj, m_data.x, m_data.u, m_data.w)
	end

	if m_data.obj isa AugmentedLagrangianCosts
		s_data.c_max = constraint_violation(
            m_data.obj.cons,
			m_data.x, m_data.u, m_data.w,
			norm_type = Inf)
	end

	return s_data.obj
end

#TODO: clean up
function Δz!(m_data::ModelData, p_data::PolicyData, s_data::SolverData)
    T = length(m_data.x)
    fill!(m_data.z, 0.0)
    for t = 1:T-1
        m_data.z[s_data.idx_u[t]] .= p_data.K[t] * m_data.z[s_data.idx_x[t]] + p_data.k[t]
        m_data.z[s_data.idx_x[t+1]] .= m_data.dyn_deriv.fx[t] * m_data.z[s_data.idx_x[t]] + m_data.dyn_deriv.fu[t] * m_data.z[s_data.idx_u[t]]
    end
end

"""
    Problem Data
"""
struct ProblemData{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,FX,FU,FW,OX,OU,OXX,OUU,OUX}
	p_data::PolicyData{N,M,NN,MM,MN,NNN,MNN}
	m_data::ModelData{T,X,U,D,FX,FU,FW,OX,OU,OXX,OUU,OUX}
	s_data::SolverData{T}
end

function problem_data(model::Model, obj::Objective; 
    w=[zeros(d.nu) for d in model])

	# allocate policy data
    p_data = policy_data(model)

    # allocate model data
    m_data = model_data(model, obj, w=w)

    # allocate solver data
    s_data = solver_data(model)

	ProblemData(p_data, m_data, s_data)
end

function nominal_trajectory(prob::ProblemData)
	return prob.m_data.x̄, prob.m_data.ū
end

function current_trajectory(prob::ProblemData)
	return prob.m_data.x, prob.m_data.u
end

#TODO: constraints
function problem_data(model::Model, obj::Objective, cons::Constraints,
    w=[zeros(d.nu) for d in dyn])

	# # constraints
	# c_data = constraints_data(model, [c.p for c in cons], T, n = n, m = m)
	# cons = StageConstraints(con_set, c_data, T)

	# # augmented Lagrangian
	# obj_al = augmented_lagrangian(obj, cons)

	# allocate policy data
    p_data = policy_data(model)

    # allocate model data
    m_data = model_data(model, obj_al, w=w)

    # allocate solver data
    s_data = solver_data(model)

	ProblemData(p_data, m_data, s_data)
end
