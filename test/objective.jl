@testset "Objective" begin
    T = 3
    nx = 2
    nu = 1 
    nw = 0
    ot = (x, u, w) -> dot(x, x) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 10.0 * dot(x, x)
    ct = Cost(ot, nx, nu, nw=nw)
    cT = Cost(oT, nx, 0, nw=nw)
    obj = [[ct for t = 1:T-1]..., cT]

    J = [0.0]
    grad = zeros((T - 1) * (nx + nu) + nx)
    indices_stateu = [collect((t - 1) * (nx + nu) .+ (1:(nx + (t == T ? 0 : nu)))) for t = 1:T]
    x1 = ones(nx) 
    u1 = ones(nu)
    w1 = zeros(nw) 
    X = [x1 for t = 1:T]
    U = [t < T ? u1 : zeros(0) for t = 1:T]
    W = [w1 for t = 1:T]

    ct.val(ct.val_cache, x1, u1, w1)
    ct.gradient_state(ct.gradient_state_cache, x1, u1, w1)
    ct.gradient_action(ct.gradient_action_cache, x1, u1, w1)

    @test ct.val_cache[1] ≈ ot(x1, u1, w1)
    @test norm(ct.gradient_state_cache - 2.0 * x1) < 1.0e-8
    @test norm(ct.gradient_action_cache - 0.2 * u1) < 1.0e-8

    cT.val(cT.val_cache, x1, u1, w1)
    cT.gradient_state(cT.gradient_state_cache, x1, zeros(0), zeros(0))
    @test cT.val_cache[1] ≈ oT(x1, u1, w1)
    @test norm(cT.gradient_state_cache - 20.0 * x1) < 1.0e-8

    @test IterativeLQR.cost(obj, X, U, X) - sum([ot(X[t], U[t], W[t]) for t = 1:T-1]) - oT(X[T], U[T], W[T]) ≈ 0.0
    gradient_state = [zeros(nx) for t = 1:T]
    gradient_action = [zeros(nu) for t = 1:T-1]
    IterativeLQR.cost_gradient!(gradient_state, gradient_action, obj, X, U, W) 
    grad = vcat([[gradient_state[t]; gradient_action[t]] for t = 1:T-1]..., gradient_state[T]...)
    @test norm(grad - vcat([[2.0 * x1; 0.2 * u1] for t = 1:T-1]..., 20.0 * x1)) < 1.0e-8
end