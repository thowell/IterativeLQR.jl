struct Dynamics{T}
    val 
    jac
    ny::Int 
    nx::Int 
    nu::Int
    nw::Int
    val_cache::Vector{T} 
    jac_cache::Matrix{T}
end

Model{T} = Vector{Dynamics{T}} where T

function Dynamics(f::Function, nx::Int, nu::Int, nw::Int=0)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu], w[1:nw] 
    y = f(x, u, w) 
    jac = Symbolics.jacobian(y, [x; u]);
    val_func = eval(Symbolics.build_function(y, x, u, w)[2]);
    jac_func = eval(Symbolics.build_function(jac, x, u, w)[2]);
    ny = length(y)
    nj, mj = size(jac)
  
    return Dynamics(val_func, jac_func, ny, nx, nu, nw, zeros(ny), zeros(nj, mj))
end

num_var(model::Model) = sum([d.nx + d.nu for d in model]) + model[end].ny

