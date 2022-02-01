@with_kw mutable struct Options{T} 
    linesearch::Symbol=:armijo
    max_iter::Int=100
    max_al_iter::Int=10
    α_min::T=1.0e-5
    obj_tol::T=1.0e-3
    grad_tol::T=1.0e-3
    con_tol::T=5.0e-3
    con_norm_type::T=Inf
    ρ_init::T=1.0
    ρ_scale::T=10.0
    ρ_max::T=1.0e8
    verbose=true
end

"""
    Problem Data
"""
mutable struct Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
	p_data::PolicyData{N,M,NN,MM,MN,NNN,MNN}
	m_data::ModelData{T,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
	s_data::SolverData{T}
    options::Options{T}
end

function solver(model::Model{T}, obj::Objective{T}; 
    w=[[zeros(d.nw) for d in model]..., zeros(0)],
    opts=Options{T}()) where T

	# allocate policy data
    p_data = policy_data(model)

    # allocate model data
    m_data = model_data(model, obj, w=w)

    # allocate solver data
    s_data = solver_data(model)

	Solver(p_data, m_data, s_data, options)
end


function get_trajectory(prob::Solver)
	return prob.m_data.x̄, prob.m_data.ū[1:end-1]
end

function current_trajectory(prob::Solver)
	return prob.m_data.x, prob.m_data.u[1:end-1]
end

function solver(model::Model{T}, obj::Objective{T}, cons::Constraints{T};
    w=[[zeros(d.nw) for d in model]..., zeros(0)],
    opts=Options{T}()) where T

	# augmented Lagrangian
	obj_al = augmented_lagrangian(model, obj, cons)

    # allocate policy data  
    p_data = policy_data(model)

    # allocate model data
    m_data = model_data(model, obj_al, w=w)

    # allocate solver data
    s_data = solver_data(model)

	Solver(p_data, m_data, s_data, opts)
end

function initialize_controls!(prob::Solver, u) 
    for (t, ut) in enumerate(u) 
        prob.m_data.ū[t] .= ut
    end 
end

function initialize_states!(prob::Solver, x) 
    for (t, xt) in enumerate(x)
        prob.m_data.x̄[t] .= xt
    end
end