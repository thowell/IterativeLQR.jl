"""
    Constraints Data
"""
struct ConstraintsData{T,C,CX,CU}
    constraints::Constraints{T}
    violations::Vector{C}
    jacobian_state::Vector{CX}
    jacobian_action::Vector{CU}
end

function constraint_data(model::Model, cons::Constraints) 
    H = length(cons)
    c = [zeros(cons[t].num_constraint) for t = 1:H]
    cx = [zeros(cons[t].num_constraint, t < H ? model[t].num_state : model[H-1].num_next_state) for t = 1:H]
    cu = [zeros(cons[t].num_constraint, model[t].num_action) for t = 1:H-1]
    ConstraintsData(cons, c, cx, cu)
end

function constraints!(constraint_data::ConstraintsData, x, u, w)
    constraints!(constraint_data.violations, constraint_data.constraints, x, u, w)
end

function constraint_violation(constraint_data::ConstraintsData; 
    norm_type=Inf)

    constraints = constraint_data.constraints
    H = length(constraints)
    max_violation = 0.0
    for t = 1:H
        num_constraint = constraints[t].num_constraint 
        ineq = constraints[t].indices_inequality
        for i = 1:num_constraint 
            c = constraint_data.violations[t][i]
            cti = (i in ineq) ? max(0.0, c) : abs(c)
            max_violation = max(max_violation, cti)
        end
    end
    return max_violation
end

function constraint_violation(constraint_data::ConstraintsData, x, u, w; 
    norm_type=Inf)
    constraints!(constraint_data, x, u, w)
    constraint_violation(constraint_data, 
        norm_type=norm_type)
end