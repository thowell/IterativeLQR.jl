@testset "Constraints" begin 
    T = 5
    num_state = 2
    num_action = 1 
    num_parameter = 0
    dim_x = [num_state for t = 1:T] 
    dim_u = [num_action for t = 1:T-1]
    dim_w = [num_parameter for t = 1:T]
    x = [rand(dim_x[t]) for t = 1:T] 
    u = [[rand(dim_u[t]) for t = 1:T-1]..., zeros(0)]
    w = [rand(dim_w[t]) for t = 1:T]

    ct = (x, u, w) -> [-ones(num_state) - x; x - ones(num_state)]
    cT = (x, u, w) -> x

    cont = Constraint(ct, num_state, num_action, indices_inequality=collect(1:2num_state), num_parameter=num_parameter)
    conT = Constraint(cT, num_state, 0, indices_inequality=collect(1:num_state), num_parameter=num_parameter)

    cons = [[cont for t = 1:T-1]..., conT]
    
    nct = 2 * num_state
    ncT = num_state
    ct0 = zeros(nct) 
    cT0 = zeros(ncT)
    cont.evaluate(ct0, x[1], u[1], w[1])
    conT.evaluate(cT0, x[T], u[T], w[T])
    @test norm(ct0 - [-ones(num_state) - x[1]; x[1] - ones(num_state)]) < 1.0e-8
    @test norm(cT0 - x[T]) < 1.0e-8

    cc = [[zeros(nct) for t = 1:T-1]..., zeros(ncT)]
    IterativeLQR.constraint!(cc, cons, x, u, w)

    @test norm(vcat(cc...) - vcat([ct(x[t], u[t], w[t]) for t = 1:T-1]..., cT(x[T], u[T], w[T]))) < 1.0e-8
    
    jx = [[zeros(nct, num_state) for t = 1:T-1]..., zeros(ncT, num_state)]
    ju = [[zeros(nct, num_action) for t = 1:T-1]..., zeros(ncT, num_action)]
    IterativeLQR.jacobian!(jx, ju, cons, x, u, w)

    for t = 1:T-1
        @test norm(jx[t] - ForwardDiff.jacobian(x -> ct(x, u[t], w[t]), x[t])) < 1.0e-8
        @test norm(ju[t] - ForwardDiff.jacobian(u -> ct(x[t], u, w[t]), u[t])) < 1.0e-8
    end
    @test norm(jx[T] - ForwardDiff.jacobian(x -> cT(x, u[T], w[T]), x[T])) < 1.0e-8
end