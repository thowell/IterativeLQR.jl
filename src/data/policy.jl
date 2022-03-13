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