@testset "Hessian of Lagrangian" begin
    const MOI = IterativeLQR.MOI
    # horizon 
    T = 3

    # acrobot 
    nx = 4 
    nu = 1 
    nw = 0 
    w_dim = [nw for t = 1:T]

    function acrobot(x, u, w)
        # dimensions
        n = 4
        m = 1
        d = 0

        # link 1
        mass1 = 1.0  
        inertia1 = 0.33  
        length1 = 1.0 
        lengthcom1 = 0.5 

        # link 2
        mass2 = 1.0  
        inertia2 = 0.33  
        length2 = 1.0 
        lengthcom2 = 0.5 

        gravity = 9.81 
        friction1 = 0.1 
        friction2 = 0.1

        # mass matrix
        function M(x, w)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

        return [a b; b c]
        end

        # dynamics bias
        function τ(x, w)
            a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
                - mass2 * gravity * (length1 * sin(x[1])
                + lengthcom2 * sin(x[1] + x[2])))

            b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

            return [a; b]
        end

        function C(x, w)
            a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
            d = 0.0

            return [a b; c d]
        end

        # input Jacobian
        function B(x, w)
            [0.0; 1.0]
        end

        # dynamics
        q = view(x, 1:2)
        v = view(x, 3:4)

        qdd = M(q, w) \ (-1.0 * C(x, w) * v
                + τ(q, w) + B(q, w) * u[1] - [friction1; friction2] .* v)

        return [x[3]; x[4]; qdd[1]; qdd[2]]
    end

    function midpoint_implicit(y, x, u, w)
        h = 0.05 # timestep 
        y - (x + h * acrobot(0.5 * (x + y), u, w))
    end

    dt = Dynamics(midpoint_implicit, nx, nx, nu, nw=nw, eval_hess=true)
    dyn = [dt for t = 1:T-1] 
    model = DynamicsModel(dyn, w_dim=w_dim)

    # initial state 
    x1 = [0.0; 0.0; 0.0; 0.0] 

    # goal state
    xT = [0.0; π; 0.0; 0.0] 

    # objective 
    ot = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4])
    objt = Cost(ot, nx, nu, nw, [t for t = 1:T-1], eval_hess=true)
    objT = Cost(oT, nx, 0, nw, [T], eval_hess=true)
    obj = [objt, objT]

    # constraints
    x_init = Bound(nx, nu, [1], xl=x1, xu=x1)
    x_goal = Bound(nx, 0, [T], xl=xT, xu=xT)
    ct = (x, u, w) -> [-5.0 * ones(nu) - cos.(u) .* sum(x.^2); cos.(x) .* tan.(u) - 5.0 * ones(nx)]
    cT = (x, u, w) -> sin.(x.^3.0)
    cont = StageConstraint(ct, nx, nu, nw, [t for t = 1:T-1], :inequality, eval_hess=true)
    conT = StageConstraint(cT, nx, 0, nw, [T], :equality, eval_hess=true)
    cons = ConstraintSet([x_init, x_goal], [cont, conT])

    # problem 
    trajopt = TrajectoryOptimizationProblem(obj, model, cons)
    s = Solver(trajopt, eval_hess=true)

    # Lagrangian
    function lagrangian(z) 
        x1 = z[1:nx] 
        u1 = z[nx .+ (1:nu)] 
        x2 = z[nx + nu .+ (1:nx)] 
        u2 = z[nx + nu + nx .+ (1:nu)] 
        x3 = z[nx + nu + nx + nu .+ (1:nx)]
        λ1_dyn = z[nx + nu + nx + nu + nx .+ (1:nx)] 
        λ2_dyn = z[nx + nu + nx + nu + nx + nx .+ (1:nx)] 

        λ1_stage = z[nx + nu + nx + nu + nx + nx + nx .+ (1:(nu + nx))] 
        λ2_stage = z[nx + nu + nx + nu + nx + nx + nx + nu + nx .+ (1:(nu + nx))] 
        λ3_stage = z[nx + nu + nx + nu + nx + nx + nx + nu + nx + nu + nx .+ (1:nx)]

        L = 0.0 
        L += ot(x1, u1, zeros(nw)) 
        L += ot(x2, u2, zeros(nw)) 
        L += oT(x3, zeros(0), zeros(nw))
        L += dot(λ1_dyn, midpoint_implicit(x2, x1, u1, zeros(nw))) 
        L += dot(λ2_dyn, midpoint_implicit(x3, x2, u2, zeros(nw))) 
        L += dot(λ1_stage, ct(x1, u1, zeros(nw)))
        L += dot(λ2_stage, ct(x2, u2, zeros(nw)))
        L += dot(λ3_stage, cT(x3, zeros(0), zeros(nw)))
        return L
    end

    nz = nx + nu + nx + nu + nx + nx + nx + nu + nx + nu + nx + nx
    np = nx + nu + nx + nu + nx
    nd = nx + nx + nu + nx + nu + nx + nx
    @variables z[1:nz]
    L = lagrangian(z)
    Lxx = Symbolics.hessian(L, z[1:np])
    Lxx_sp = Symbolics.sparsehessian(L, z[1:np])
    spar = [findnz(Lxx_sp)[1:2]...]
    Lxx_func = eval(Symbolics.build_function(Lxx, z)[1])
    Lxx_sp_func = eval(Symbolics.build_function(Lxx_sp.nzval, z)[1])

    z0 = rand(nz)
    nh = length(s.p.sp_hess_lag)
    h0 = zeros(nh)

    σ = 1.0
    fill!(h0, 0.0)
    IterativeLQR.trajectory!(s.p.trajopt.x, s.p.trajopt.u, z0[1:np], 
        s.p.trajopt.model.idx.x, s.p.trajopt.model.idx.u)
    IterativeLQR.duals!(s.p.trajopt.λ_dyn, s.p.trajopt.λ_stage, z0[np .+ (1:nd)], s.p.idx.dyn_con, s.p.idx.stage_con)
    IterativeLQR.eval_obj_hess!(h0, s.p.idx.obj_hess, s.p.trajopt.obj, s.p.trajopt.x, s.p.trajopt.u, s.p.trajopt.w, σ)
    IterativeLQR.eval_hess_lag!(h0, s.p.idx.dyn_hess, s.p.trajopt.model.dyn, s.p.trajopt.x, s.p.trajopt.u, s.p.trajopt.w, s.p.trajopt.λ_dyn)
    IterativeLQR.eval_hess_lag!(h0, s.p.idx.stage_hess, s.p.trajopt.con.stage, s.p.trajopt.x, s.p.trajopt.u, s.p.trajopt.w, s.p.trajopt.λ_stage)

    sp_obj_hess = IterativeLQR.sparsity_hessian(obj, model.dim.x, model.dim.u)
    sp_dyn_hess = IterativeLQR.sparsity_hessian(dyn, model.dim.x, model.dim.u)
    sp_con_hess = IterativeLQR.sparsity_hessian(cons.stage, model.dim.x, model.dim.u)
    sp_hess = collect([sp_obj_hess..., sp_dyn_hess..., sp_con_hess...]) 
    sp_key = sort(unique(sp_hess))

    idx_obj_hess = IterativeLQR.hessian_indices(obj, sp_key, model.dim.x, model.dim.u)
    idx_dyn_hess = IterativeLQR.hessian_indices(dyn, sp_key, model.dim.x, model.dim.u)
    idx_con_hess = IterativeLQR.hessian_indices(cons.stage, sp_key, model.dim.x, model.dim.u)

    # indices
    @test sp_key[vcat(idx_obj_hess...)] == sp_obj_hess
    @test sp_key[vcat(idx_dyn_hess...)] == sp_dyn_hess
    @test sp_key[vcat(idx_con_hess...)] == sp_con_hess

    # Hessian
    h0_full = zeros(np, np)
    for (i, h) in enumerate(h0)
        h0_full[sp_key[i][1], sp_key[i][2]] = h
    end
    @test norm(h0_full - Lxx_func(z0)) < 1.0e-8
    @test norm(norm(h0 - Lxx_sp_func(z0))) < 1.0e-8

    h0 = zeros(nh)
    MOI.eval_hessian_lagrangian(s.p, h0, z0[1:np], σ, z0[np .+ (1:nd)])
    @test norm(norm(h0 - Lxx_sp_func(z0))) < 1.0e-8

    # a = z0[1:np]
    # b = z0[np .+ (1:nd)]
    # info = @benchmark MOI.eval_hessian_lagrangian($s.p, $h0, $a, $σ, $b)
end
