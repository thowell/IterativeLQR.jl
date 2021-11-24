function model_derivatives!(data::ModelData; mode=:nominal)
    model = data.model 
    x, u, w = trajectories(data, mode=mode)
    jx = data.model_deriv.fx
    ju = data.model_deriv.fu
    eval_con_jac!(jx, ju, model, x, u, w)
end

function objective_derivatives!(obj::Objective, data::ModelData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    gradx = data.obj_deriv.gx
    gradu = data.obj_deriv.gu
    hessxx = data.obj_deriv.gxx
    hessuu = data.obj_deriv.guu
    hessux = data.obj_deriv.gux
    eval_obj_grad!(gradx, gradu, obj, x, u, w)
    eval_obj_hess!(hessxx, hessuu, hessux, obj, x, u, w) 
end

function constraints_derivatives!(c_data::ConstraintsData, m_data::ModelData; mode=:nominal)
    x, u, w = trajectories(m_data, mode=mode)
    cx = c_data.cx 
    cu = c_data.cu
    eval_con_jac!(cx, cu, c_data.cons, x, u, w)
end

function objective_derivatives!(obj::AugmentedLagrangianCosts, data::ModelData; mode=:nominal)
    # objective 
    gx = data.obj_deriv.gx
    gu = data.obj_deriv.gu
    gxx = data.obj_deriv.gxx
    guu = data.obj_deriv.guu
    gux = data.obj_deriv.gux

    # constraints
    c = obj.c_data.c
    cx = obj.c_data.cx
    cu = obj.c_data.cu
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a

    T = length(obj)

    # derivatives
    objective_derivatives!(obj.costs, data, mode=mode)
    constraints_derivatives!(obj.c_data, data, mode=mode)

    # 
    for t = 1:T
        gx[t] .+= cx[t]' * (λ[t] + ρ[t] .* a[t] .* c[t])
        gxx[t] .+= cx[t]' * Diagonal(ρ[t] .* a[t]) * cx[t]
        t == T && continue 
        gu[t] .+= cu[t]' * (λ[t] + ρ[t] .* a[t] .* c[t])
        guu[t] .+= cu[t]' * Diagonal(ρ[t] .* a[t]) * cu[t]
        gux[t] .+= cu[t]' * Diagonal(ρ[t] .* a[t]) * cx[t]
    end
end

function derivatives!(m_data::ModelData; mode=:nominal)
    model_derivatives!(m_data, mode=mode)
    objective_derivatives!(m_data.obj, m_data, mode=mode)
end
