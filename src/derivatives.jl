function derivatives!(dynamics::Model, data::ModelData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    jx = data.model_deriv.fx
    ju = data.model_deriv.fu
    eval_con_jac!(jx, ju, dynamics, x, u, w)
end

function derivatives!(obj::Objective, data::ModelData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    gradx = data.obj_deriv.gx
    gradu = data.obj_deriv.gu
    hessxx = data.obj_deriv.gxx
    hessuu = data.obj_deriv.guu
    hessux = data.obj_deriv.gux
    eval_obj_grad!(gradx, gradu, obj, x, u, w)
    eval_obj_hess!(hessxx, hessuu, hessux, obj, x, u, w) 
end

function derivatives!(obj::AugmentedLagrangianCosts, data::ModelData; mode=:nominal)
    # objective 
    gx = data.obj_deriv.gx
    gu = data.obj_deriv.gu
    gxx = data.obj_deriv.gxx
    guu = data.obj_deriv.guu
    gux = data.obj_deriv.gux

    # constraints
    cons = obj.c_data.cons
    c = obj.c_data.c
    cx = obj.c_data.cx
    cu = obj.c_data.cu
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a
    Iρ = obj.Iρ
    c_tmp = obj.c_tmp 
    cx_tmp = obj.cx_tmp 
    cu_tmp = obj.cu_tmp

    T = length(obj)

    # derivatives
    derivatives!(obj.costs, data, mode=mode)
    derivatives!(obj.c_data, data, mode=mode)

    for t = 1:T
        nc = cons[t].nc
        for i = 1:nc 
            Iρ[t][i, i] = ρ[t][i] * a[t][i]
        end
        c_tmp[t] .= λ[t] 

        # gx
        mul!(c_tmp[t], Iρ[t], c[t], 1.0, 1.0)
        mul!(gx[t], transpose(cx[t]), c_tmp[t], 1.0, 1.0)

        # gxx 
        mul!(cx_tmp[t], Iρ[t], cx[t])
        mul!(gxx[t], transpose(cx[t]), cx_tmp[t], 1.0, 1.0)

        t == T && continue 

        # gu 
        mul!(gu[t], transpose(cu[t]), c_tmp[t], 1.0, 1.0) 

        # guu 
        mul!(cu_tmp[t], Iρ[t], cu[t]) 
        mul!(guu[t], transpose(cu[t]), cu_tmp[t], 1.0, 1.0) 

        # gux 
        mul!(gux[t], transpose(cu[t]), cx_tmp[t], 1.0, 1.0)
        
        # gx[t] .+= cx[t]' * (λ[t] + ρ[t] .* a[t] .* c[t])
        # gxx[t] .+= cx[t]' * Diagonal(ρ[t] .* a[t]) * cx[t]
        # t == T && continue 
        # gu[t] .+= cu[t]' * (λ[t] + ρ[t] .* a[t] .* c[t])
        # guu[t] .+= cu[t]' * Diagonal(ρ[t] .* a[t]) * cu[t]
        # gux[t] .+= cu[t]' * Diagonal(ρ[t] .* a[t]) * cx[t]
    end
end

function derivatives!(c_data::ConstraintsData, m_data::ModelData; mode=:nominal)
    x, u, w = trajectories(m_data, mode=mode)
    cx = c_data.cx 
    cu = c_data.cu
    eval_con_jac!(cx, cu, c_data.cons, x, u, w)
end

function derivatives!(m_data::ModelData; 
    mode=:nominal)
    derivatives!(m_data.model_deriv.dynamics, m_data, mode=mode)
    derivatives!(m_data.obj_deriv.costs, m_data, mode=mode)
end
