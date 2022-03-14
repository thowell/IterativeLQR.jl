mutable struct AugmentedLagrangianCosts{T,C,CX,CU}
    costs::Objective{T}
    constraint_data::ConstraintsData{T,C,CX,CU}
    constraint_penalty::Vector{Vector{T}} 
    constraint_penalty_matrix::Vector{Diagonal{T,Vector{T}}}              
    constraint_dual::Vector{Vector{T}}               
    active_set::Vector{Vector{Int}}            
    constraint_tmp::Vector{Vector{T}}
    constraint_jacobian_state_tmp::Vector{Matrix{T}} 
    constraint_jacobian_action_tmp::Vector{Matrix{T}}
end

function augmented_lagrangian(model::Model{T}, costs::Objective{T}, constraints::Constraints{T}) where T
    # horizon
    H = length(model) + 1
    # penalty
    constraint_penalty = [ones(c.num_constraint) for c in constraints]
    constraint_penalty_matrix = [Diagonal(ones(c.num_constraint)) for c in constraints]
    # duals
    constraint_dual = [zeros(c.num_constraint) for c in constraints]
    # active set
    active_set = [ones(Int, c.num_constraint) for c in constraints]
    # pre-allocated memory
    constraint_tmp = [zeros(c.num_constraint) for c in constraints]
    constraint_jacobian_state_tmp = [zeros(c.num_constraint, t < H ? model[t].num_state : model[H-1].num_next_state) for (t, c) in enumerate(constraints)]
    constraint_jacobian_action_tmp = [zeros(c.num_constraint, t < H ? model[t].num_action : 0) for (t, c) in enumerate(constraints)]
    data = constraint_data(model, constraints)
    AugmentedLagrangianCosts(costs, 
        data, 
        constraint_penalty,
        constraint_penalty_matrix,  
        constraint_dual, 
        active_set, 
        constraint_tmp, 
        constraint_jacobian_state_tmp, 
        constraint_jacobian_action_tmp)
end

function cost(obj::AugmentedLagrangianCosts, states, actions, parameters)
    # costs
    J = cost(obj.costs, states, actions, parameters)

    # constraints
    c = obj.constraint_data.violations
    ρ = obj.constraint_penalty
    λ = obj.constraint_dual
    a = obj.active_set

    # horizon
    H = length(c)

    constraints!(obj.constraint_data, states, actions, parameters)
    active_set!(a, obj.constraint_data, λ)

    for t = 1:H
        J += λ[t]' * c[t]
        num_constraint = obj.constraint_data.constraints[t].num_constraint 
        for i = 1:num_constraint 
            if a[t][i] == 1
                J += 0.5 * ρ[t][i] * c[t][i]^2.0
            end
        end
    end

    return J
end

function active_set!(a, data::ConstraintsData, λ)
    # violations
    c = data.violations
    
    # horizon
    H = length(c)

    for t = 1:H
        # set all constraints active
        fill!(a[t], 1)

        # check inequality constraints
        for i in data.constraints[t].indices_inequality
            # check active-set criteria
            (c[t][i] < 0.0 && λ[t][i] == 0.0) && (a[t][i] = 0)
        end
    end
end

function augmented_lagrangian_update!(obj::AugmentedLagrangianCosts;
        scaling_penalty=10.0, 
        max_penalty=1.0e12)

    # constraints
    c = obj.constraint_data.violations
    cons = obj.constraint_data.constraints
    ρ = obj.constraint_penalty
    λ = obj.constraint_dual

    # horizon
    H = length(c)

    for t = 1:H
        num_constraint = cons[t].num_constraint 
        for i = 1:num_constraint 
            λ[t][i] += ρ[t][i] * c[t][i]
            if i in cons[t].indices_inequality
                λ[t][i] = max(0.0, λ[t][i])
            end
            ρ[t][i] = min(scaling_penalty * ρ[t][i], max_penalty)
        end
    end
end

Base.length(obj::AugmentedLagrangianCosts) = length(obj.costs)