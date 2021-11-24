mutable struct AugmentedLagrangianCosts{T,C,CX,CU}
    costs::Objective{T}
    c_data::ConstraintsData{T,C,CX,CU}
    ρ::Vector{Vector{T}}               # penalty
    λ::Vector{Vector{T}}               # dual estimates
    a::Vector{Vector{Int}}             # active set
end

function augmented_lagrangian(model::Model{T}, costs::Objective{T}, cons::Constraints{T}) where T
    ρ = [ones(c.nc) for c in cons]
    λ = [zeros(c.nc) for c in cons]
    a = [ones(Int, c.nc) for c in cons]
    c_data = constraints_data(model, cons)
    AugmentedLagrangianCosts(costs, c_data, ρ, λ, a)
end

function eval_obj(obj::AugmentedLagrangianCosts, x, u, w)
    # costs
    J = eval_obj(obj.costs, x, u, w)

    # constraints
    c = obj.c_data.c
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a
    T = length(c)

    constraints!(obj.c_data, x, u, w)
    active_set!(a, obj.c_data, λ)

    for t = 1:T
        J += λ[t]' * c[t]
        J += 0.5 * c[t]' * Diagonal(ρ[t] .* a[t]) * c[t]
    end

    return J
end

function active_set!(a, c_data::ConstraintsData, λ)
    c = c_data.c
    T = length(c)

    for t = 1:T
        # set all constraints active
        fill!(a[t], 1)

        # check inequality constraints
        idx = c_data.cons[t].idx_ineq
        if length(idx) > 0
            for i in idx
                # check active-set criteria
                @show (c[t][i] < 0.0 && λ[t][i] == 0.0) && (a[t][i] = 0)
            end
        end
    end
end

function augmented_lagrangian_update!(obj::AugmentedLagrangianCosts;
        s = 10.0, max_penalty = 1.0e12)
    # constraints
    c = obj.c_data.c
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a
    T = length(c)

    for t = 1:T
        # dual estimate update
        λ[t] .+= ρ[t] .* c[t]

        # inequality projection
        idx = obj.c_data.cons[t].idx_ineq
        if length(idx) > 0
            λ[t][idx] = max.(0.0, view(λ[t], idx))
        end

        ρ[t] .= min.(s .* ρ[t], max_penalty)
    end
end

Base.length(obj::AugmentedLagrangianCosts) = length(obj.costs)