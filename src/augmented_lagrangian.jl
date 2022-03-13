mutable struct AugmentedLagrangianCosts{T,C,CX,CU}
    costs::Objective{T}
    constraint_data::ConstraintsData{T,C,CX,CU}
    ρ::Vector{Vector{T}}               # penalty
    λ::Vector{Vector{T}}               # dual estimates
    a::Vector{Vector{Int}}             # active set
    Iρ::Vector{Diagonal{T,Vector{T}}}
    c_tmp::Vector{Vector{T}}
    cx_tmp::Vector{Matrix{T}} 
    cu_tmp::Vector{Matrix{T}}
end

function augmented_lagrangian(model::Model{T}, costs::Objective{T}, cons::Constraints{T}) where T
    H = length(model) + 1
    ρ = [ones(c.nc) for c in cons]
    λ = [zeros(c.nc) for c in cons]
    a = [ones(Int, c.nc) for c in cons]
    Iρ = [Diagonal(ones(c.nc)) for c in cons]
    c_tmp = [zeros(c.nc) for c in cons]
    cx_tmp = [zeros(c.nc, t < H ? model[t].nx : model[H-1].ny) for (t, c) in enumerate(cons)]
    cu_tmp = [zeros(c.nc, t < H ? model[t].nu : 0) for (t, c) in enumerate(cons)]
    data = constraint_data(model, cons)
    AugmentedLagrangianCosts(costs, data, ρ, λ, a, Iρ, 
        c_tmp, cx_tmp, cu_tmp)
end

function cost(obj::AugmentedLagrangianCosts, x, u, w)
    # costs
    J = cost(obj.costs, x, u, w)

    # constraints
    c = obj.constraint_data.c
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a
    T = length(c)

    constraints!(obj.constraint_data, x, u, w)
    active_set!(a, obj.constraint_data, λ)

    for t = 1:T
        J += λ[t]' * c[t]
        nc = obj.constraint_data.cons[t].nc 
        for i = 1:nc 
            if a[t][i] == 1
                J += 0.5 * ρ[t][i] * c[t][i]^2.0
            end
        end
    end
    return J
end

function active_set!(a, constraint_data::ConstraintsData, λ)
    c = constraint_data.c
    T = length(c)

    for t = 1:T
        # set all constraints active
        fill!(a[t], 1)

        # check inequality constraints
        for i in constraint_data.cons[t].idx_ineq
            # check active-set criteria
            (c[t][i] < 0.0 && λ[t][i] == 0.0) && (a[t][i] = 0)
        end
    end
end

function augmented_lagrangian_update!(obj::AugmentedLagrangianCosts;
        s = 10.0, max_penalty = 1.0e12)
    # constraints
    c = obj.constraint_data.c
    cons = obj.constraint_data.cons
    ρ = obj.ρ
    λ = obj.λ
    T = length(c)

    for t = 1:T
        nc = cons[t].nc 
        for i = 1:nc 
            λ[t][i] += ρ[t][i] * c[t][i]
            if i in cons[t].idx_ineq
                λ[t][i] = max(0.0, λ[t][i])
            end
            ρ[t][i] = min(s * ρ[t][i], max_penalty)
        end
    end
end

Base.length(obj::AugmentedLagrangianCosts) = length(obj.costs)