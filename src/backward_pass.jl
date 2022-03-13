function backward_pass!(policy::PolicyData, problem::ProblemData; mode = :nominal)

    T = length(problem.x)
    fx = problem.model.fx
    fu = problem.model.fu
    gx = problem.objective.gx
    gu = problem.objective.gu
    gxx = problem.objective.gxx
    guu = problem.objective.guu
    gux = problem.objective.gux

    # policy
    if mode == :nominal
        K = policy.K
        k = policy.k
    else
        K = policy.K_cand
        k = policy.k_cand
    end

    # value function approximation
    P = policy.P
    p = policy.p

    # state-action value function approximation
    Qx = policy.Qx
    Qu = policy.Qu
    Qxx = policy.Qxx
    Quu = policy.Quu
    Qux = policy.Qux

    # terminal value function
    P[T] .= gxx[T]
    p[T] .=  gx[T]

    for t = T-1:-1:1
        # Qx[t] .= gx[t] + fx[t]' * p[t+1]
        mul!(Qx[t], transpose(fx[t]), p[t+1])
        Qx[t] .+= gx[t]

        # Qu[t] .= gu[t] + fu[t]' * p[t+1]
        mul!(Qu[t], transpose(fu[t]), p[t+1])
        Qu[t] .+= gu[t]

        # Qxx[t] .= gxx[t] + fx[t]' * P[t+1] * fx[t]
        mul!(policy.xx̂_tmp[t], transpose(fx[t]), P[t+1])
        mul!(Qxx[t], policy.xx̂_tmp[t], fx[t])
        Qxx[t] .+= gxx[t]

        # Quu[t] .= guu[t] + fu[t]' * P[t+1] * fu[t]
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), P[t+1])
        mul!(Quu[t], policy.ux̂_tmp[t], fu[t])
        Quu[t] .+= guu[t]

        # Qux[t] .= gux[t] + fu[t]' * P[t+1] * fx[t]
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), P[t+1])
        mul!(Qux[t], policy.ux̂_tmp[t], fx[t])
        Qux[t] .+= gux[t]

        # K[t] .= -1.0 * Quu[t] \ Qux[t]
        # k[t] .= -1.0 * Quu[t] \ Qu[t]
		policy.uu_tmp[t] .= Quu[t]
        LAPACK.potrf!('U', policy.uu_tmp[t])
        K[t] .= Qux[t]
        k[t] .= Qu[t]
        LAPACK.potrs!('U', policy.uu_tmp[t], K[t])
		LAPACK.potrs!('U', policy.uu_tmp[t], k[t])
		K[t] .*= -1.0
		k[t] .*= -1.0

        # P[t] .=  Qxx[t] + K[t]' * Quu[t] * K[t] + K[t]' * Qux[t] + Qux[t]' * K[t]
        # p[t] .=  Qx[t] + K[t]' * Quu[t] * k[t] + K[t]' * Qu[t] + Qux[t]' * k[t]
		mul!(policy.ux_tmp[t], Quu[t], K[t])

		mul!(P[t], transpose(K[t]), policy.ux_tmp[t])
		mul!(P[t], transpose(K[t]), Qux[t], 1.0, 1.0)
		mul!(P[t], transpose(Qux[t]), K[t], 1.0, 1.0)
		P[t] .+= Qxx[t]

		mul!(p[t], transpose(policy.ux_tmp[t]), k[t])
		mul!(p[t], transpose(K[t]), Qu[t], 1.0, 1.0)
		mul!(p[t], transpose(Qux[t]), k[t], 1.0, 1.0)
		p[t] .+= Qx[t]
    end
end
