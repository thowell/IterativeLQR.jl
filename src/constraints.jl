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
    idx_ineq::Vector{Int}
end

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(f::Function, nx::Int, nu::Int; 
    idx_ineq::Vector{Int}=collect(1:0), nw::Int=0)

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
        idx_ineq)
end

function Constraint()
    return Constraint(
        (c, x, u, w) -> nothing, 
        (jx, x, u, w) -> nothing, (ju, x, u, w) -> nothing, 
        0, 0, 0, 0, 
        Float64[], Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), 
        collect(1:0))
end

function eval_con!(c, cons::Constraints{T}, x, u, w) where T
    for (t, con) in enumerate(cons)
        con.nc == 0 && continue
        con.val(con.val_cache, x[t], u[t], w[t])
        @views c[t] .= con.val_cache
        fill!(con.val_cache, 0.0) # TODO: confirm this is necessary 
    end
end

function eval_con_jac!(jacx, jacu, cons::Constraints{T}, x, u, w) where T
    H = length(cons)
    for (t, con) in enumerate(cons[1:H-1])
        con.nc == 0 && continue
        con.jacx(con.jacx_cache, x[t], u[t], w[t])
        con.jacu(con.jacu_cache, x[t], u[t], w[t])
        @views jacx[t] .= con.jacx_cache
        @views jacu[t] .= con.jacu_cache
        fill!(con.jacx_cache, 0.0) # TODO: confirm this is necessary
        fill!(con.jacu_cache, 0.0) # TODO: confirm this is necessary
    end
    cons[H].nc == 0 && return
    cons[H].jacx(cons[H].jacx_cache, x[H], u[H], w[H])
    @views jacx[H] .= cons[H].jacx_cache
    fill!(cons[H].jacx_cache, 0.0) # TODO: confirm this is necessary
end

"""
    Constraints Data
"""
struct ConstraintsData{T,C,CX,CU}
    c::Vector{C}
    cx::Vector{CX}
    cu::Vector{CU}
    cons::Constraints{T}
end

function constraints_data(model::Model, cons::Constraints) 
    T = length(cons)
    c = [zeros(cons[t].nc) for t = 1:T]
    cx = [zeros(cons[t].nc, t < T ? model[t].nx : model[T-1].ny) for t = 1:T]
    cu = [zeros(cons[t].nc, model[t].nu) for t = 1:T-1]
    ConstraintsData(c, cx, cu, cons)
end

function constraints!(c_data::ConstraintsData, x, u, w)
    eval_con!(c_data.c, c_data.cons, x, u, w)
end

function constraint_violation(c_data::ConstraintsData; norm_type=Inf)
    T = length(c_data.cons)
    c_max = 0.0
    for t = 1:T
        c_viol = copy(c_data.c[t]) # TODO: may be unnecessary

        # project inequality constraints
        for i in c_data.cons[t].idx_ineq
            c_viol[i] = max.(0.0, c_data.c[t][i])
        end

        c_max = max(c_max, norm(c_viol, norm_type))
    end
    return c_max
end

function constraint_violation(c_data::ConstraintsData, x, u, w; norm_type=Inf)
    constraints!(c_data, x, u, w)
    constraint_violation(c_data, norm_type=norm_type)
end