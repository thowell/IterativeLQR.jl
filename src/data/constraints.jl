"""
    Constraints Data
"""
struct ConstraintsData{T,C,CX,CU}
    c::Vector{C}
    cx::Vector{CX}
    cu::Vector{CU}
    cons::Constraints{T}
end

function constraint_data(model::Model, cons::Constraints) 
    T = length(cons)
    c = [zeros(cons[t].nc) for t = 1:T]
    cx = [zeros(cons[t].nc, t < T ? model[t].nx : model[T-1].ny) for t = 1:T]
    cu = [zeros(cons[t].nc, model[t].nu) for t = 1:T-1]
    ConstraintsData(c, cx, cu, cons)
end

function constraints!(constraint_data::ConstraintsData, x, u, w)
    constraints!(constraint_data.c, constraint_data.cons, x, u, w)
end

function constraint_violation(constraint_data::ConstraintsData; 
    norm_type=Inf)
    cons = constraint_data.cons
    T = length(cons)
    c_max = 0.0
    for t = 1:T
        nc = cons[t].nc 
        ineq = cons[t].idx_ineq
        for i = 1:nc 
            cti = (i in ineq) ? max(0.0, constraint_data.c[t][i]) : abs(constraint_data.c[t][i])
            c_max = max(c_max, cti)
        end
    end
    return c_max
end

function constraint_violation(constraint_data::ConstraintsData, x, u, w; 
    norm_type=Inf)
    constraints!(constraint_data, x, u, w)
    constraint_violation(constraint_data, norm_type=norm_type)
end