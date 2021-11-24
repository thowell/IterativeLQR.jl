struct Constraint{T}
    val 
    jacx 
    jacu
    nc::Int
    nx::Int 
    nu::Int 
    nw::Int
    val_cache::Vector{T} 
    jacx_cache::Matrix{T}
    jacu_cache::Matrix{T}
    type::Symbol
end

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(f::Function, nx::Int, nu::Int, type::Symbol, nw::Int=0)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu], w[1:nw]
    
    val = f(x, u, w)
    jacx = Symbolics.jacobian(val, x)
    jacu = Symbolics.jacobian(val, u)

    val_func = eval(Symbolics.build_function(val, x, u, w)[2])
    jacx_func = eval(Symbolics.build_function(jacx, x, u, w)[2])
    jacu_func = eval(Symbolics.build_function(jacu, x, u, w)[2])

    nc = length(val) 
    
    return Constraint(
        val_func, 
        jacx_func, jacu_func,
        nc, nx, nu, nw,  
        zeros(nc), zeros(nc, nx), zeros(nc, nu), 
        type)
end

function Constraint()
    return Constraint(
        (c, x, u, w) -> nothing, 
        (jx, x, u, w) -> nothing, (ju, x, u, w) -> nothing, 
        0, 0, 0, 0, 
        Float64[], Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), 
        :empty)
end

function eval_con!(c, idx, cons::Constraints{T}, x, u, w) where T
    for (t, con) in enumerate(cons)
        con.val(con.val_cache, x[t], u[t], w[t])
        @views c[t] .= con.val_cache
        fill!(con.val_cache, 0.0) # TODO: confirm this is necessary 
    end
end

function eval_con_jac!(jacx, jacu, cons::Constraints{T}, x, u, w) where T
    for (t, con) in enumerate(cons)
        con.jacx(con.jacx_cache, x[t], u[t], w[t])
        con.jacu(con.jacu_cache, x[t], u[t], w[t])
        @views jacx[t] .= con.jacx_cache
        @views jacu[t] .= con.jacu_cache
        fill!(con.jacx_cache, 0.0) # TODO: confirm this is necessary
        fill!(con.jacu_cache, 0.0) # TODO: confirm this is necessary
    end
end

"""
    Constraints Data
"""
struct ConstraintsData
    c
    cx
    cu
end

function constraints_data(model::Model, p::Vector, T::Int;
	n = [model.n for t = 1:T],
	m = [model.m for t = 1:T-1])

    c = [zeros(p[t]) for t = 1:T]
    cx = [zeros(p[t], n[t]) for t = 1:T]
    cu = [zeros(p[t], m[t]) for t = 1:T-1]
    ConstraintsData(c, cx, cu)
end

# struct StageConstraint
#     p::Int
#     info::Dict
# end

# ConstraintSet = Vector{StageConstraint}

# struct StageConstraints <: Constraints
#     con::ConstraintSet
#     data::ConstraintsData
#     T::Int
# end

# c!(a, cons::StageConstraints, x, u, t) = nothing

# function constraints!(cons::StageConstraints, x, u)
#     T = cons.T

#     for t = 1:T-1
#         c!(cons.data.c[t], cons, x[t], u[t], t)
#     end

#     c!(cons.data.c[T], cons, x[T], nothing, T)
# end

# function constraint_violation(cons::StageConstraints; norm_type = Inf)
#     T = cons.T
#     c_max = 0.0

#     for t = 1:T
#         c_viol = copy(cons.data.c[t])

#         # find inequality constraints
#         if haskey(cons.con[t].info, :inequality)
#             for i in cons.con[t].info[:inequality]
#                 c_viol[i] = max.(0.0, cons.data.c[t][i])
#             end
#         end

#         c_max = max(c_max, norm(c_viol, norm_type))
#     end

#     return c_max
# end

# function constraint_violation(cons::StageConstraints, x, u; norm_type = Inf)
#     constraints!(cons, x, u)
#     constraint_violation(cons, norm_type = norm_type)
# end