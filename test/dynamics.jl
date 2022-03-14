@testset "Dynamics" begin 
    T = 3
    nx = 2 
    nu = 1 
    nw = 0 
    w_dim = [nw for t = 1:T]
    
    function pendulum(z, u, w) 
        mass = 1.0 
        lc = 1.0 
        gravity = 9.81 
        damping = 0.1
        [z[2], (u[1] / ((mass * lc * lc)) - gravity * sin(z[1]) / lc - damping * z[2] / (mass * lc * lc))]
    end

    function euler_explicit(x, u, w)
        h = 0.1
        x + h * pendulum(x, u, w)
    end

    dyn = Dynamics(euler_explicit, nx, nu, nw=nw)
    model = [dyn for t = 1:T-1]

    x1 = ones(nx) 
    u1 = ones(nu)
    w1 = zeros(nw)
    X = [x1 for t = 1:T]
    U = [u1 for t = 1:T]
    W = [w1 for t = 1:T]

    dyn.val(dyn.val_cache, x1, u1, w1) 
    @test norm(dyn.val_cache - euler_explicit(x1, u1, w1)) < 1.0e-8
    dyn.jacobian_state(dyn.jacobian_state_cache, x1, u1, w1) 
    dyn.jacobian_action(dyn.jacobian_action_cache, x1, u1, w1) 

    jac_fd = ForwardDiff.jacobian(a -> euler_explicit(a[1:nx], a[nx .+ (1:nu)], w1), [x1; u1])
    @test norm([dyn.jacobian_state_cache dyn.jacobian_action_cache] - jac_fd) < 1.0e-8

    d = [zeros(nx) for t = 1:T-1]
    for (t, dyn) in enumerate(model) 
        d[t] .= dynamics!(dyn, X[t], U[t], W[t])
    end
    @test norm(vcat(d...) - vcat([euler_explicit(X[t], U[t], W[t]) for t = 1:T-1]...)) < 1.0e-8
  
    jx = [zeros(nx, nx) for t = 1:T-1] 
    ju = [zeros(nx, nu) for t = 1:T-1]
    IterativeLQR.jacobian!(jx, ju, model, X, U, W) 
    jac_dense = [jx[1] ju[1] zeros(nx, nx + nu);
                 zeros(nx, nx + nu) jx[2] ju[2]]
    @test norm(jac_dense - [jac_fd zeros(model[2].nx, model[2].nu + model[2].ny); zeros(model[2].ny, model[1].nx + model[1].nu) jac_fd]) < 1.0e-8
end

