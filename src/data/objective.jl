struct ObjectiveData{C,X,U,XX,UU,UX}
    costs::C
    gx::Vector{X}
    gu::Vector{U}
    gxx::Vector{XX}
    guu::Vector{UU}
    gux::Vector{UX}
end

function objective_data(dynamics::Model{T}, costs) where T
	gx = [[zeros(d.nx) for d in dynamics]..., 
        zeros(dynamics[end].ny)]
    gu = [zeros(d.nu) for d in dynamics]
    gxx = [[zeros(d.nx, d.nx) for d in dynamics]..., 
        zeros(dynamics[end].ny, dynamics[end].ny)]
    guu = [zeros(d.nu, d.nu) for d in dynamics]
    gux = [zeros(d.nu, d.nx) for d in dynamics]
    ObjectiveData(costs, gx, gu, gxx, guu, gux)
end

function reset!(data::ObjectiveData) 
    T = length(data.gx) 
    for t = 1:T 
        fill!(data.gx[t], 0.0) 
        fill!(data.gxx[t], 0.0) 
        t == T && continue
        fill!(data.gu[t], 0.0)
        fill!(data.guu[t], 0.0)
        fill!(data.gux[t], 0.0)
    end 
end