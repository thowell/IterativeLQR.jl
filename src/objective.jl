struct Cost{T}
    #TODO: types for methods
    val
    gradx 
    gradu
    hessxx 
    hessuu 
    hessux
    val_cache::Vector{T}
    gradx_cache::Vector{T}
    gradu_cache::Vector{T}
    hessxx_cache::Matrix{T}
    hessuu_cache::Matrix{T}
    hessux_cache::Matrix{T}
end

function Cost(f::Function, nx::Int, nu::Int, nw::Int=0)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu], w[1:nw]
    
    val = f(x, u, w)
    gradx = Symbolics.gradient(val, x)
    gradu = Symbolics.gradient(val, u) 
    hessxx = Symbolics.jacobian(gradx, x) 
    hessuu = Symbolics.jacobian(gradu, u) 
    hessux = Symbolics.jacobian(gradu, x) 

    val_func = eval(Symbolics.build_function([val], x, u, w)[2])
    gradx_func = eval(Symbolics.build_function(gradx, x, u, w)[2])
    gradu_func = eval(Symbolics.build_function(gradu, x, u, w)[2])
    hessxx_func = eval(Symbolics.build_function(hessxx, x, u, w)[2])
    hessuu_func = eval(Symbolics.build_function(hessuu, x, u, w)[2])
    hessux_func = eval(Symbolics.build_function(hessux, x, u, w)[2])  

    return Cost(val_func, 
        gradx_func, gradu_func, 
        hessxx_func, hessuu_func, hessux_func,
        zeros(1), 
        zeros(nx), zeros(nu), 
        zeros(nx, nx), zeros(nu, nu), zeros(nu, nx))
end

Objective{T} = Vector{Cost{T}} where T

function eval_obj(obj::Objective, x, u, w) 
    J = 0.0
    for (t, cost) in enumerate(obj)
        cost.val(cost.val_cache, x[t], u[t], w[t])
        J += cost.val_cache[1]
    end
    return J 
end

function eval_obj_grad!(gradx, gradu, obj::Objective, x, u, w)
    T = length(obj)
    for (t, cost) in enumerate(obj[1:end-1])
        cost.gradx(cost.gradx_cache, x[t], u[t], w[t])
        cost.gradu(cost.gradu_cache, x[t], u[t], w[t])
        @views gradx[t] .= cost.gradx_cache
        @views gradu[t] .= cost.gradu_cache
        fill!(cost.gradx_cache, 0.0) # TODO: confirm this is necessary
        fill!(cost.gradu_cache, 0.0) # TODO: confirm this is necessary
    end
    obj[T].gradx(obj[T].gradx_cache, x[T], u[T], w[T])
    @views gradx[T] .= obj[T].gradx_cache
    fill!(obj[T].gradx_cache, 0.0) # TODO: confirm this is necessary
end

function eval_obj_hess!(hessxx, hessuu, hessux, obj::Objective, x, u, w)
    T = length(obj) 
    for (t, cost) in enumerate(obj[1:T-1])
        cost.hessxx(cost.hessxx_cache, x[t], u[t], w[t])
        cost.hessuu(cost.hessuu_cache, x[t], u[t], w[t])
        cost.hessux(cost.hessux_cache, x[t], u[t], w[t])
        @views hessxx[t] .+= cost.hessxx_cache
        @views hessuu[t] .+= cost.hessuu_cache
        @views hessux[t] .+= cost.hessux_cache
        fill!(cost.hessxx_cache, 0.0) # TODO: confirm this is necessary
        fill!(cost.hessuu_cache, 0.0) # TODO: confirm this is necessary
        fill!(cost.hessux_cache, 0.0) # TODO: confirm this is necessary
    end
    obj[T].hessxx(obj[T].hessxx_cache, x[T], u[T], w[T])
    @views hessxx[T] .+= obj[T].hessxx_cache
    fill!(obj[T].hessxx_cache, 0.0) # TODO: confirm this is necessary
end