@testset "Solve: acrobot" begin 
    # ## horizon 
    T = 51 

    # ## car 
    nx = 3
    nu = 2
    nw = 0 

    function car(x, u, w)
        [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
    end

    function midpoint_explicit(x, u, w)
        h = 0.1 # timestep 
        x + h * car(x + 0.5 * h * car(x, u, w), u, w)
    end

    # ## model
    dyn = Dynamics(midpoint_explicit, nx, nu, nw)
    model = [dyn for t = 1:T-1] 

    # ## initialization
    x1 = [0.0; 0.0; 0.0] 
    xT = [1.0; 1.0; 0.0] 

    # ## rollout
    ū = [1.0e-2 * [1.0; 0.1] for t = 1:T-1]
    w = [zeros(nw) for t = 1:T-1] 
    x̄ = rollout(model, x1, ū, w)

    # ## objective 
    ot = (x, u, w) -> 1.0 * dot(x - xT, x - xT) + 1.0e-2 * dot(u, u)
    oT = (x, u, w) -> 1000.0 * dot(x - xT, x - xT)
    ct = Cost(ot, nx, nu, nw)
    cT = Cost(oT, nx, 0, nw)
    obj = [[ct for t = 1:T-1]..., cT]

    # ## constraints
    ul = -5.0 * ones(nu) 
    uu = 5.0 * ones(nu)

    p_obs = [0.5; 0.5] 
    r_obs = 0.1

    function stage_con(x, u, w) 
        e = x[1:2] - p_obs
        [
        ul - u; # control limit (lower)
        u - uu; # control limit (upper)
        r_obs^2.0 - dot(e, e); # obstacle 
        ]
    end 

    function terminal_con(x, u, w) 
        e = x[1:2] - p_obs
        [
        x - xT; # goal 
        r_obs^2.0 - dot(e, e); # obstacle
        ]
    end

    cont = Constraint(stage_con, nx, nu, idx_ineq=collect(1:5))
    conT = Constraint(terminal_con, nx, nu, idx_ineq=collect(3 .+ (1:1)))
    cons = [[cont for t = 1:T-1]..., conT] 

    # ## problem
    prob = problem_data(model, obj, cons)
    initialize_controls!(prob, ū) 
    initialize_states!(prob, x̄)

    # ## solve
    constrained_ilqr_solve!(prob, 
        linesearch=:armijo,
        verbose=false,
        α_min=1.0e-5,
        obj_tol=1.0e-3,
        grad_tol=1.0e-3,
        max_iter=100,
        max_al_iter=10,
        con_tol=1.0e-3,
        ρ_init=1.0,
        ρ_scale=10.0)

    # ## solution
    x_sol, u_sol = nominal_trajectory(prob)

    @test all([all(stage_con(x_sol[t], u_sol[t], w[t]) .<= 1.0e-3) for t = 1:T-1])
    @test all(terminal_con(x_sol[T], zeros(0), zeros(0)) .<= 1.0e-3)
end