mutable struct AugmentedLagrangianCosts{T}
    costs::Objective{T}
    cons::Constraints{T}
    ρ::Vector{Vector{T}}  # penalty
    λ::Vector{Vector{T}}  # dual estimates
    a::Vector{Vector{Int}}  # active set
end

function augmented_lagrangian(costs::Objective, cons::Constraints)
    λ = [zeros(c.p) for c in cons]
    a = [ones(Int, c.p) for c in cons]
    ρ = [ones(c.p) for c in cons]
    AugmentedLagrangianCosts(costs, cons, ρ, λ, a)
end

function objective(obj::AugmentedLagrangianCosts, x, u)
    # costs
    J = objective(obj.costs, x, u)

    # constraints
    T = obj.cons.T
    c = obj.cons.data.c
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a

    constraints!(obj.cons, x, u)
    active_set!(a, obj.cons, λ)

    for t = 1:T
        J += λ[t]' * c[t]
        J += 0.5 * c[t]' * Diagonal(ρ[t] .* a[t]) * c[t]
    end

    return J
end

function active_set!(a, cons::Constraints, λ)
    T = cons.T
    c = cons.data.c

    for t = 1:T
        # set all constraints active
        fill!(a[t], 1.0)

        # find inequality constraints
        if haskey(cons.con[t].info, :inequality)
            for i in cons.con[t].info[:inequality]
                # check active-set criteria
                (c[t][i] < 0.0 && λ[t][i] == 0.0) && (a[t][i] = 0.0)
            end
        end
    end
end

function augmented_lagrangian_update!(obj::AugmentedLagrangianCosts;
        s = 10.0, max_penalty = 1.0e12)
    # constraints
    T = obj.cons.T
    c = obj.cons.data.c
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a

    for t = 1:T
        # dual estimate update
        λ[t] .+= ρ[t] .* c[t]

        # inequality projection
        if haskey(obj.cons.con[t].info, :inequality)
            idx = obj.cons.con[t].info[:inequality]
            λ[t][idx] = max.(0.0, view(λ[t], idx))
        end
        ρ[t] .= min.(s .* ρ[t], max_penalty)
    end
end
