""" 
    Value function approximation 
"""
struct Value{N,NN}
    gradient::Vector{N}
    hessian::Vector{NN}
end

""" 
    Action-value function approximation 
"""
struct ActionValue{N,M,NN,MM,MN}
    gradient_state::Vector{N}
    gradient_action::Vector{M}
    hessian_state_state::Vector{NN}
    hessian_action_action::Vector{MM}
    hessian_action_state::Vector{MN}
end

"""
    Policy Data
"""
struct PolicyData{N,M,NN,MM,MN,NNN,MNN}
    # policy u = ū + K * (x - x̄) + k
    K::Vector{MN}
    k::Vector{M}

    K_candidate::Vector{MN}
    k_candidate::Vector{M}

    # value function approximation
    value::Value{N,NN}

    # action-value function approximation
    action_value::ActionValue{N,M,NN,MM,MN}

    # pre-allocated memory
	xx̂_tmp::Vector{NNN}
	ux̂_tmp::Vector{MNN}
	uu_tmp::Vector{MM}
	ux_tmp::Vector{MN}
end

function policy_data(dynamics::Vector{Dynamics{T}}) where T
    # policy
	K = [zeros(d.nu, d.nx) for d in dynamics]
    k = [zeros(d.nu) for d in dynamics]

    K_candidate = [zeros(d.nu, d.nx) for d in dynamics]
    k_candidate = [zeros(d.nu) for d in dynamics]

    # value function approximation
    P = [[zeros(d.nx, d.nx) for d in dynamics]..., 
            zeros(dynamics[end].ny, dynamics[end].ny)]
    p =  [[zeros(d.nx) for d in dynamics]..., 
            zeros(dynamics[end].ny)]

    value = Value(p, P)

    # action-value function approximation
    Qx = [zeros(d.nx) for d in dynamics]
    Qu = [zeros(d.nu) for d in dynamics]
    Qxx = [zeros(d.nx, d.nx) for d in dynamics]
    Quu = [zeros(d.nu, d.nu) for d in dynamics]
    Qux = [zeros(d.nu, d.nx) for d in dynamics]

    action_value = ActionValue(Qx, Qu, Qxx, Quu, Qux)

	xx̂_tmp = [zeros(d.nx, d.ny) for d in dynamics]
	ux̂_tmp = [zeros(d.nu, d.ny) for d in dynamics]
	uu_tmp = [zeros(d.nu, d.nu) for d in dynamics]
	ux_tmp = [zeros(d.nu, d.nx) for d in dynamics]

    PolicyData(K, k, K_candidate, k_candidate, 
        value,
        action_value,
        xx̂_tmp, ux̂_tmp, uu_tmp, ux_tmp)
end