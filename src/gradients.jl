function gradients!(dynamics::Model, data::ProblemData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    jx = data.model.fx
    ju = data.model.fu
    jacobian!(jx, ju, dynamics, x, u, w)
end

function gradients!(obj::Objective, data::ProblemData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    gradx = data.objective.gx
    gradu = data.objective.gu
    hessxx = data.objective.gxx
    hessuu = data.objective.guu
    hessux = data.objective.gux
    cost_gradient!(gradx, gradu, obj, x, u, w)
    cost_hessian!(hessxx, hessuu, hessux, obj, x, u, w) 
end

function gradients!(obj::AugmentedLagrangianCosts, data::ProblemData; mode=:nominal)
    # objective 
    gx = data.objective.gx
    gu = data.objective.gu
    gxx = data.objective.gxx
    guu = data.objective.guu
    gux = data.objective.gux

    # constraints
    cons = obj.constraint_data.cons
    c = obj.constraint_data.c
    cx = obj.constraint_data.cx
    cu = obj.constraint_data.cu
    ρ = obj.ρ
    λ = obj.λ
    a = obj.a
    Iρ = obj.Iρ
    c_tmp = obj.c_tmp 
    cx_tmp = obj.cx_tmp 
    cu_tmp = obj.cu_tmp

    T = length(obj)

    # derivatives
    gradients!(obj.costs, data, mode=mode)
    gradients!(obj.constraint_data, data, mode=mode)

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

function gradients!(constraint_data::ConstraintsData, problem::ProblemData; mode=:nominal)
    x, u, w = trajectories(problem, mode=mode)
    cx = constraint_data.cx 
    cu = constraint_data.cu
    jacobian!(cx, cu, constraint_data.cons, x, u, w)
end

function gradients!(problem::ProblemData; 
    mode=:nominal)
    gradients!(problem.model.dynamics, problem, mode=mode)
    gradients!(problem.objective.costs, problem, mode=mode)
end
