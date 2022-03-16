@testset "Dynamics" begin 
    T = 3
    num_state = 2 
    num_action = 1 
    num_parameter = 0 
    w_dim = [num_parameter for t = 1:T]
    
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

    dynamics = Dynamics(euler_explicit, num_state, num_action, num_parameter=num_parameter)
    model = [dynamics for t = 1:T-1]

    x1 = ones(num_state) 
    u1 = ones(num_action)
    w1 = zeros(num_parameter)
    X = [x1 for t = 1:T]
    U = [u1 for t = 1:T]
    W = [w1 for t = 1:T]

    dynamics.evaluate(dynamics.evaluate_cache, x1, u1, w1) 
    @test norm(dynamics.evaluate_cache - euler_explicit(x1, u1, w1)) < 1.0e-8
    dynamics.jacobian_state(dynamics.jacobian_state_cache, x1, u1, w1) 
    dynamics.jacobian_action(dynamics.jacobian_action_cache, x1, u1, w1) 

    jac_fd = ForwardDiff.jacobian(a -> euler_explicit(a[1:num_state], a[num_state .+ (1:num_action)], w1), [x1; u1])
    @test norm([dynamics.jacobian_state_cache dynamics.jacobian_action_cache] - jac_fd) < 1.0e-8

    d = [zeros(num_state) for t = 1:T-1]
    for (t, dynamics) in enumerate(model) 
        d[t] .= dynamics!(dynamics, X[t], U[t], W[t])
    end
    @test norm(vcat(d...) - vcat([euler_explicit(X[t], U[t], W[t]) for t = 1:T-1]...)) < 1.0e-8
  
    jx = [zeros(num_state, num_state) for t = 1:T-1] 
    ju = [zeros(num_state, num_action) for t = 1:T-1]
    IterativeLQR.jacobian!(jx, ju, model, X, U, W) 
    jac_dense = [jx[1] ju[1] zeros(num_state, num_state + num_action);
                 zeros(num_state, num_state + num_action) jx[2] ju[2]]
    @test norm(jac_dense - [jac_fd zeros(model[2].num_state, model[2].num_action + model[2].num_next_state); zeros(model[2].num_next_state, model[1].num_state + model[1].num_action) jac_fd]) < 1.0e-8
end

