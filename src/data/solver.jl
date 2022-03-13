"""
    Solver Data
"""
struct SolverData{T}
    obj::Vector{T}              # objective value
    gradient::Vector{T}         # Lagrangian gradient
	c_max::Vector{T}            # maximum constraint violation

    idx_x::Vector{Vector{Int}}  # indices for state trajectory
    idx_u::Vector{Vector{Int}}  # indices for control trajectory

    α::Vector{T}                # step length
    status::Vector{Bool}        # solver status

    iter::Vector{Int}

	cache::Dict{Symbol,Vector{T}}  # solver stats
end

function solver_data(model::Model{T}; max_cache=1000) where T
    # indices x and u
    idx_x = Vector{Int}[]
    idx_u = Vector{Int}[] 
    n_sum = 0 
    m_sum = 0 
    n_total = sum([d.nx for d in model]) + model[end].ny
    for d in model
        push!(idx_x, collect(n_sum .+ (1:d.nx))) 
        push!(idx_u, collect(n_total + m_sum .+ (1:d.nu)))
        n_sum += d.nx 
        m_sum += d.nu 
    end
    push!(idx_x, collect(n_sum .+ (1:model[end].ny)))

    obj = [Inf]
	c_max = [0.0]
    α = [1.0]
    gradient = zeros(num_var(model))
	cache = Dict(:obj => zeros(max_cache), 
                 :grad => zeros(max_cache), 
                 :c_max => zeros(max_cache), 
                 :α => zeros(max_cache))

    SolverData(obj, gradient, c_max, idx_x, idx_u, α, [false], [0], cache)
end

function reset!(data::SolverData) 
    fill!(data.obj, 0.0) 
    fill!(data.gradient, 0.0)
    fill!(data.c_max, 0.0) 
    fill!(data.cache[:obj], 0.0) 
    fill!(data.cache[:grad], 0.0) 
    fill!(data.cache[:c_max], 0.0) 
    fill!(data.cache[:α], 0.0) 
    data.status[1] = false
    data.iter[1] = 0
end

# TODO: fix iter
function cache!(data::SolverData)
    iter = 1 #data.cache[:iter] 
    # (iter > length(data[:obj])) && (@warn "solver data cache exceeded")
	data.cache[:obj][iter] = data.obj
	data.cache[:gradient][iter] = data.gradient
	data.cache[:c_max][iter] = data.c_max
	data.cache[:α][iter] = data.α
    return nothing
end