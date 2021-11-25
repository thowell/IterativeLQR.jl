@testset "Objective" begin
    T = 3
    nx = 2
    nu = 1 
    nw = 0
    ot = (x, u, w) -> dot(x, x) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 10.0 * dot(x, x)
    ct = Cost(ot, nx, nu, nw)
    cT = Cost(oT, nx, 0, nw)
    obj = [[ct for t = 1:T-1]..., cT]

    J = [0.0]
    grad = zeros((T - 1) * (nx + nu) + nx)
    idx_xu = [collect((t - 1) * (nx + nu) .+ (1:(nx + (t == T ? 0 : nu)))) for t = 1:T]
    x1 = ones(nx) 
    u1 = ones(nu)
    w1 = zeros(nw) 
    X = [x1 for t = 1:T]
    U = [t < T ? u1 : zeros(0) for t = 1:T]
    W = [w1 for t = 1:T]

    ct.val(ct.val_cache, x1, u1, w1)
    ct.gradx(ct.gradx_cache, x1, u1, w1)
    ct.gradu(ct.gradu_cache, x1, u1, w1)

    @test ct.val_cache[1] ≈ ot(x1, u1, w1)
    @test norm(ct.gradx_cache - 2.0 * x1) < 1.0e-8
    @test norm(ct.gradu_cache - 0.2 * u1) < 1.0e-8

    cT.val(cT.val_cache, x1, u1, w1)
    cT.gradx(cT.gradx_cache, x1, zeros(0), zeros(0))
    @test cT.val_cache[1] ≈ oT(x1, u1, w1)
    @test norm(cT.gradx_cache - 20.0 * x1) < 1.0e-8

    @test IterativeLQR.eval_obj(obj, X, U, X) - sum([ot(X[t], U[t], W[t]) for t = 1:T-1]) - oT(X[T], U[T], W[T]) ≈ 0.0
    gradx = [zeros(nx) for t = 1:T]
    gradu = [zeros(nu) for t = 1:T-1]
    IterativeLQR.eval_obj_grad!(gradx, gradu, obj, X, U, W) 
    grad = vcat([[gradx[t]; gradu[t]] for t = 1:T-1]..., gradx[T]...)
    @test norm(grad - vcat([[2.0 * x1; 0.2 * u1] for t = 1:T-1]..., 20.0 * x1)) < 1.0e-8
end