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
    c = [zeros(cons[t].nc) for t = 1:H]
    cx = [zeros(cons[t].nc, t < H ? model[t].nx : model[H-1].ny) for t = 1:H]
    cu = [zeros(cons[t].nc, model[t].nu) for t = 1:H-1]
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
        nc = constraints[t].nc 
        ineq = constraints[t].indices_inequality
        for i = 1:nc 
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