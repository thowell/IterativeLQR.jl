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

function constraints!(c, cons::Constraints{T}, x, u, w) where T
    for (t, con) in enumerate(cons)
        con.nc == 0 && continue
        con.val(con.val_cache, x[t], u[t], w[t])
        @views c[t] .= con.val_cache
        fill!(con.val_cache, 0.0) # TODO: confirm this is necessary 
    end
end

function jacobian!(jacx, jacu, cons::Constraints{T}, x, u, w) where T
    H = length(cons)
    for (t, con) in enumerate(cons)
        con.nc == 0 && continue
        con.jacx(con.jacx_cache, x[t], u[t], w[t])
        @views jacx[t] .= con.jacx_cache
        fill!(con.jacx_cache, 0.0) # TODO: confirm this is necessary
        t == H && continue
        con.jacu(con.jacu_cache, x[t], u[t], w[t])
        @views jacu[t] .= con.jacu_cache
        fill!(con.jacu_cache, 0.0) # TODO: confirm this is necessary
    end
end
