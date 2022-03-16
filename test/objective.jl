@testset "Objective" begin
    T = 3
    num_state = 2
    num_action = 1 
    num_parameter = 0
    ot = (x, u, w) -> dot(x, x) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 10.0 * dot(x, x)
    ct = Cost(ot, num_state, num_action, num_parameter=num_parameter)
    cT = Cost(oT, num_state, 0, num_parameter=num_parameter)
    obj = [[ct for t = 1:T-1]..., cT]

    J = [0.0]
    grad = zeros((T - 1) * (num_state + num_action) + num_state)
    indices_stateu = [collect((t - 1) * (num_state + num_action) .+ (1:(num_state + (t == T ? 0 : num_action)))) for t = 1:T]
    x1 = ones(num_state) 
    u1 = ones(num_action)
    w1 = zeros(num_parameter) 
    X = [x1 for t = 1:T]
    U = [t < T ? u1 : zeros(0) for t = 1:T]
    W = [w1 for t = 1:T]

    ct.evaluate(ct.evaluate_cache, x1, u1, w1)
    ct.gradient_state(ct.gradient_state_cache, x1, u1, w1)
    ct.gradient_action(ct.gradient_action_cache, x1, u1, w1)

    @test ct.evaluate_cache[1] ≈ ot(x1, u1, w1)
    @test norm(ct.gradient_state_cache - 2.0 * x1) < 1.0e-8
    @test norm(ct.gradient_action_cache - 0.2 * u1) < 1.0e-8

    cT.evaluate(cT.evaluate_cache, x1, u1, w1)
    cT.gradient_state(cT.gradient_state_cache, x1, zeros(0), zeros(0))
    @test cT.evaluate_cache[1] ≈ oT(x1, u1, w1)
    @test norm(cT.gradient_state_cache - 20.0 * x1) < 1.0e-8

    @test IterativeLQR.cost(obj, X, U, X) - sum([ot(X[t], U[t], W[t]) for t = 1:T-1]) - oT(X[T], U[T], W[T]) ≈ 0.0
    gradient_state = [zeros(num_state) for t = 1:T]
    gradient_action = [zeros(num_action) for t = 1:T-1]
    IterativeLQR.cost_gradient!(gradient_state, gradient_action, obj, X, U, W) 
    grad = vcat([[gradient_state[t]; gradient_action[t]] for t = 1:T-1]..., gradient_state[T]...)
    @test norm(grad - vcat([[2.0 * x1; 0.2 * u1] for t = 1:T-1]..., 20.0 * x1)) < 1.0e-8
end