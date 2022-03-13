@testset "Constraints" begin 
    T = 5
    nx = 2
    nu = 1 
    nw = 0
    dim_x = [nx for t = 1:T] 
    dim_u = [nu for t = 1:T-1]
    dim_w = [nw for t = 1:T]
    x = [rand(dim_x[t]) for t = 1:T] 
    u = [[rand(dim_u[t]) for t = 1:T-1]..., zeros(0)]
    w = [rand(dim_w[t]) for t = 1:T]

    ct = (x, u, w) -> [-ones(nx) - x; x - ones(nx)]
    cT = (x, u, w) -> x

    cont = Constraint(ct, nx, nu, idx_ineq=collect(1:2nx), nw=nw)
    conT = Constraint(cT, nx, 0, idx_ineq=collect(1:nx), nw=nw)

    cons = [[cont for t = 1:T-1]..., conT]
    
    nct = 2 * nx
    ncT = nx
    ct0 = zeros(nct) 
    cT0 = zeros(ncT)
    cont.val(ct0, x[1], u[1], w[1])
    conT.val(cT0, x[T], u[T], w[T])
    @test norm(ct0 - [-ones(nx) - x[1]; x[1] - ones(nx)]) < 1.0e-8
    @test norm(cT0 - x[T]) < 1.0e-8

    cc = [[zeros(nct) for t = 1:T-1]..., zeros(ncT)]
    IterativeLQR.constraints!(cc, cons, x, u, w)

    @test norm(vcat(cc...) - vcat([ct(x[t], u[t], w[t]) for t = 1:T-1]..., cT(x[T], u[T], w[T]))) < 1.0e-8
    
    jx = [[zeros(nct, nx) for t = 1:T-1]..., zeros(ncT, nx)]
    ju = [[zeros(nct, nu) for t = 1:T-1]..., zeros(ncT, nu)]
    IterativeLQR.jacobian!(jx, ju, cons, x, u, w)

    for t = 1:T-1
        @test norm(jx[t] - ForwardDiff.jacobian(x -> ct(x, u[t], w[t]), x[t])) < 1.0e-8
        @test norm(ju[t] - ForwardDiff.jacobian(u -> ct(x[t], u, w[t]), u[t])) < 1.0e-8
    end
    @test norm(jx[T] - ForwardDiff.jacobian(x -> cT(x, u[T], w[T]), x[T])) < 1.0e-8
end