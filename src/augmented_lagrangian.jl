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

function augmented_lagrangian(model::Model{T}, costs::Objective{T}, cons::Constraints{T}) where T
    H = length(model) + 1
    constraint_penalty = [ones(c.nc) for c in cons]
    constraint_dual = [zeros(c.nc) for c in cons]
    active_set = [ones(Int, c.nc) for c in cons]
    constraint_penalty_matrix = [Diagonal(ones(c.nc)) for c in cons]
    constraint_tmp = [zeros(c.nc) for c in cons]
    constraint_jacobian_state_tmp = [zeros(c.nc, t < H ? model[t].nx : model[H-1].ny) for (t, c) in enumerate(cons)]
    constraint_jacobian_action_tmp = [zeros(c.nc, t < H ? model[t].nu : 0) for (t, c) in enumerate(cons)]
    data = constraint_data(model, cons)
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

function cost(obj::AugmentedLagrangianCosts, x, u, w)
    # costs
    J = cost(obj.costs, x, u, w)

    # constraints
    violations = obj.constraint_data.violations
    constraint_penalty = obj.constraint_penalty
    constraint_dual = obj.constraint_dual
    active_set = obj.active_set
    T = length(violations)

    constraints!(obj.constraint_data, x, u, w)
    active_set!(active_set, obj.constraint_data, constraint_dual)

    for t = 1:T
        J += constraint_dual[t]' * violations[t]
        nc = obj.constraint_data.constraints[t].nc 
        for i = 1:nc 
            if active_set[t][i] == 1
                J += 0.5 * constraint_penalty[t][i] * violations[t][i]^2.0
            end
        end
    end
    return J
end

function active_set!(active_set, constraint_data::ConstraintsData, constraint_dual)
    violations = constraint_data.violations
    T = length(violations)

    for t = 1:T
        # set all constraints active
        fill!(active_set[t], 1)

        # check inequality constraints
        for i in constraint_data.constraints[t].idx_ineq
            # check active-set criteria
            (violations[t][i] < 0.0 && constraint_dual[t][i] == 0.0) && (active_set[t][i] = 0)
        end
    end
end

function augmented_lagrangian_update!(obj::AugmentedLagrangianCosts;
        scaling_penalty=10.0, 
        max_penalty=1.0e12)
    # constraints
    violations = obj.constraint_data.violations
    constraints = obj.constraint_data.constraints
    constraint_penalty = obj.constraint_penalty
    constraint_dual = obj.constraint_dual
    T = length(violations)

    for t = 1:T
        nc = constraints[t].nc 
        for i = 1:nc 
            constraint_dual[t][i] += constraint_penalty[t][i] * violations[t][i]
            if i in constraints[t].idx_ineq
                constraint_dual[t][i] = max(0.0, constraint_dual[t][i])
            end
            constraint_penalty[t][i] = min(scaling_penalty * constraint_penalty[t][i], max_penalty)
        end
    end
end

Base.length(obj::AugmentedLagrangianCosts) = length(obj.costs)