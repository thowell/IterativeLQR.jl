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
    T = length(cons)
    c = [zeros(cons[t].nc) for t = 1:T]
    cx = [zeros(cons[t].nc, t < T ? model[t].nx : model[T-1].ny) for t = 1:T]
    cu = [zeros(cons[t].nc, model[t].nu) for t = 1:T-1]
    ConstraintsData(cons, c, cx, cu)
end

function constraints!(constraint_data::ConstraintsData, x, u, w)
    constraints!(constraint_data.violations, constraint_data.constraints, x, u, w)
end

function constraint_violation(constraint_data::ConstraintsData; 
    norm_type=Inf)

    constraints = constraint_data.constraints
    T = length(constraints)
    c_max = 0.0
    for t = 1:T
        nc = constraints[t].nc 
        ineq = constraints[t].idx_ineq
        for i = 1:nc 
            cti = (i in ineq) ? max(0.0, constraint_data.violations[t][i]) : abs(constraint_data.violations[t][i])
            c_max = max(c_max, cti)
        end
    end
    return c_max
end

function constraint_violation(constraint_data::ConstraintsData, x, u, w; 
    norm_type=Inf)
    constraints!(constraint_data, x, u, w)
    constraint_violation(constraint_data, 
        norm_type=norm_type)
end