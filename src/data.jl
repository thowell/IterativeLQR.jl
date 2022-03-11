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

function reset!(data::ModelDerivativesData) 
    T = length(data.fx) + 1
    for t = 1:T-1 
        fill!(data.fx[t], 0.0) 
        fill!(data.fu[t], 0.0) 
        # fill!(data.fw[t], 0.0) 
    end 
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

function reset!(data::ObjectiveDerivativesData) 
    T = length(data.gx) 
    for t = 1:T 
        fill!(data.gx[t], 0.0) 
        fill!(data.gxx[t], 0.0) 
        t == T && continue
        fill!(data.gu[t], 0.0)
        fill!(data.guu[t], 0.0)
        fill!(data.gux[t], 0.0)
    end 
end

mutable struct ModelData{T,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
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
    obj::O

    # dynamics derivatives data
    model_deriv::ModelDerivativesData{FX,FU,FW}

    # objective derivatives data
    obj_deriv::ObjectiveDerivativesData{OX,OU,OXX,OUU,OUX}

    # z = (x1...,xT,u1,...,uT-1) | Δz = (Δx1...,ΔxT,Δu1,...,ΔuT-1)
    z::Vector{T}
end

function model_data(model::Model, obj; 
    w=[[zeros(d.nw) for d in model]..., zeros(0)])

    length(w) == length(model) && (w = [w..., zeros(0)])
    @assert length(model) + 1 == length(w)
    @assert length(model) + 1 == length(obj)

	x = [[zeros(d.nx) for d in model]..., 
            zeros(model[end].ny)]
    u = [[zeros(d.nu) for d in model]..., zeros(0)]

    x̄ = [[zeros(d.nx) for d in model]..., 
            zeros(model[end].ny)]
    ū = [[zeros(d.nu) for d in model]..., zeros(0)]

    model_deriv = model_derivatives_data(model)
    obj_deriv = objective_derivatives_data(model)

    z = zeros(num_var(model))

    ModelData(x, u, w, x̄, ū, model, obj, model_deriv, obj_deriv, z)
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

function update_nominal_trajectory!(data::ModelData) 
    T = length(data.x) 
    for t = 1:T 
        data.x̄[t] .= data.x[t] 
        t == T && continue 
        data.ū[t] .= data.u[t] 
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

    iter::Vector{Int}

	cache::Dict{Symbol,Vector{T}}  # solver stats
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
                 :grad => zeros(max_cache), 
                 :c_max => zeros(max_cache), 
                 :α => zeros(max_cache))

    SolverData(obj, gradient, c_max, idx_x, idx_u, α, [false], [0], cache)
end

function reset!(data::SolverData) 
    fill!(data.obj, 0.0) 
    fill!(data.gradient, 0.0)
    fill!(data.c_max, 0.0) 
    fill!(data.cache[:obj], 0.0) 
    fill!(data.cache[:grad], 0.0) 
    fill!(data.cache[:c_max], 0.0) 
    fill!(data.cache[:α], 0.0) 
    data.status[1] = false
    data.iter[1] = 0
end

# TODO: fix iter
function cache!(data::SolverData)
    iter = 1 #data.cache[:iter] 
    # (iter > length(data[:obj])) && (@warn "solver data cache exceeded")
	data.cache[:obj][iter] = data.obj
	data.cache[:gradient][iter] = data.gradient
	data.cache[:c_max][iter] = data.c_max
	data.cache[:α][iter] = data.α
    return nothing
end

function objective!(s_data::SolverData, m_data::ModelData; mode = :nominal)
	if mode == :nominal
		s_data.obj[1] = eval_obj(m_data.obj, m_data.x̄, m_data.ū, m_data.w)
	elseif mode == :current
		s_data.obj[1] = eval_obj(m_data.obj, m_data.x, m_data.u, m_data.w)
	end

	if m_data.obj isa AugmentedLagrangianCosts
		s_data.c_max[1] = constraint_violation(
            m_data.obj.c_data,
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
        zx = @views m_data.z[s_data.idx_x[t]]
        zu = @views m_data.z[s_data.idx_u[t]]
        zy = @views m_data.z[s_data.idx_x[t+1]]
        zu .= p_data.k[t] 
        mul!(zu, p_data.K[t], zx, 1.0, 1.0)
        mul!(zy, m_data.model_deriv.fu[t], zu)
        mul!(zy, m_data.model_deriv.fx[t], zx, 1.0, 1.0)
        # m_data.z[s_data.idx_u[t]] .= p_data.K[t] * m_data.z[s_data.idx_x[t]] + p_data.k[t]
        # m_data.z[s_data.idx_x[t+1]] .= m_data.model_deriv.fx[t] * m_data.z[s_data.idx_x[t]] + m_data.model_deriv.fu[t] * m_data.z[s_data.idx_u[t]]
    end
end

function trajectories(m_data::ModelData; mode=:nominal) 
    x = mode == :nominal ? m_data.x̄ : m_data.x 
    u = mode == :nominal ? m_data.ū : m_data.u 
    w = m_data.w
    return x, u, w 
end