function model_derivatives!(data::ModelData; mode=:nominal)
    model = data.model 
    x = mode == :nominal ? data.x̄ : data.x 
    u = mode == :nominal ? data.ū : data.u 
    w = data.w
    jx = data.model_deriv.fx
    ju = data.model_deriv.fu
    eval_con_jac!(jx, ju, model, x, u, w)
end

function objective_derivatives!(data::ModelData; mode=:nominal)
    obj = data.obj 
    x = mode == :nominal ? data.x̄ : data.x 
    u = mode == :nominal ? data.ū : data.u 
    w = data.w 
    gradx = data.obj_deriv.gx
    gradu = data.obj_deriv.gu
    hessxx = data.obj_deriv.gxx
    hessuu = data.obj_deriv.guu
    hessux = data.obj_deriv.gux
    eval_obj_grad!(gradx, gradu, obj, x, u, w)
    eval_obj_hess!(hessxx, hessuu, hessux, obj, x, u, w) 
end

function constraints_derivatives!(cons::Constraints, data::ModelData;
    mode = :nominal)

    if mode == :nominal
        x̄ = data.x̄
        ū = data.ū
    else
        x̄ = data.x
        ū = data.u
    end

    # T = data.T

    # for t = 1:T-1
    #     c = cons.data.c[t]
    #     cx!(a, z) = c!(a, cons, z, ū[t], t)
    #     cu!(a, z) = c!(a, cons, x̄[t], z, t)

    #     ForwardDiff.jacobian!(cons.data.cx[t], cx!, c, x̄[t])
    #     ForwardDiff.jacobian!(cons.data.cu[t], cu!, c, ū[t])
    # end

    # c = cons.data.c[T]
    # cxT!(a, z) = c!(a, cons, z, nothing, T)
    # ForwardDiff.jacobian!(cons.data.cx[T], cxT!, c, x̄[T])
end

function objective_derivatives!(obj::AugmentedLagrangianCosts, data::ModelData;
        mode = :nominal)

    # gx = data.obj_deriv.gx
    # gu = data.obj_deriv.gu
    # gxx = data.obj_deriv.gxx
    # guu = data.obj_deriv.guu
    # gux = data.obj_deriv.gux

    # c = obj.cons.data.c
    # cx = obj.cons.data.cx
    # cu = obj.cons.data.cu
    # ρ = obj.ρ
    # λ = obj.λ
    # a = obj.a

    # T = data.T
    # model = data.model

    # objective_derivatives!(obj.costs, data, mode = mode)
    # constraints_derivatives!(obj.cons, data, mode = mode)

    # for t = 1:T-1
    #     gx[t] .+= cx[t]' * (λ[t] + ρ[t] .* a[t] .* c[t])
    #     gu[t] .+= cu[t]' * (λ[t] + ρ[t] .* a[t] .* c[t])
    #     gxx[t] .+= cx[t]' * Diagonal(ρ[t] .* a[t]) * cx[t]
    #     guu[t] .+= cu[t]' * Diagonal(ρ[t] .* a[t]) * cu[t]
    #     gux[t] .+= cu[t]' * Diagonal(ρ[t] .* a[t]) * cx[t]
    # end

    # gx[T] .+= cx[T]' * (λ[T] + ρ[T] .* a[T] .* c[T])
    # gxx[T] .+= cx[T]' * Diagonal(ρ[T] .* a[T]) * cx[T]
end

function derivatives!(m_data::ModelData; mode=:nominal)
    model_derivatives!(m_data, mode=mode)
    objective_derivatives!(m_data, mode=mode)
end
