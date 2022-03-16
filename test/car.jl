@testset "Solve: car" begin 
    # ## horizon 
    T = 51 

    # ## car 
    num_state = 3
    num_action = 2
    num_parameter = 0 

    function car(x, u, w)
        [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
    end

    function midpoint_explicit(x, u, w)
        h = 0.1 # timestep 
        x + h * car(x + 0.5 * h * car(x, u, w), u, w)
    end

    # ## model
    dyn = Dynamics(midpoint_explicit, num_state, num_action, num_parameter=num_parameter)
    model = [dyn for t = 1:T-1] 

    # ## initialization
    x1 = [0.0; 0.0; 0.0] 
    xT = [1.0; 1.0; 0.0] 

    # ## rollout
    ū = [1.0e-2 * [1.0; 0.1] for t = 1:T-1]
    w = [zeros(num_parameter) for t = 1:T] 
    x̄ = rollout(model, x1, ū, w)

    # ## objective 
    ot = (x, u, w) -> 1.0 * dot(x - xT, x - xT) + 1.0e-2 * dot(u, u)
    oT = (x, u, w) -> 0.0 * dot(x - xT, x - xT)
    ct = Cost(ot, num_state, num_action, num_parameter=num_parameter)
    cT = Cost(oT, num_state, 0, num_parameter=0)
    obj = [[ct for t = 1:T-1]..., cT]

    # ## constraints
    ul = -5.0 * ones(num_action) 
    uu = 5.0 * ones(num_action)

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

    cont = Constraint(stage_con, num_state, num_action, indices_inequality=collect(1:5))
    conT = Constraint(terminal_con, num_state, 0, indices_inequality=collect(3 .+ (1:1)))
    cons = [[cont for t = 1:T-1]..., conT] 

    # ## problem
    options = Options(verbose=false,
        line_search=:armijo,
        min_step_size=1.0e-5,
        objective_tolerance=1.0e-3,
        lagrangian_gradient_tolerance=1.0e-3,
        max_iterations=100,
        max_dual_updates=10,
        initial_constraint_penalty=1.0,
        scaling_penalty=10.0)
    s = Solver(model, obj, cons, 
        options=options)
    initialize_controls!(s, ū) 
    initialize_states!(s, x̄)

    # ## solve
    solve!(s)

    # ## solution
    x_sol, u_sol = get_trajectory(s)

    @test all([all(stage_con(x_sol[t], u_sol[t], w[t]) .<= options.constraint_tolerance) for t = 1:T-1])
    @test all(abs.(terminal_con(x_sol[T], zeros(0), zeros(0))[1:3]) .<= options.constraint_tolerance)

    # ## allocations
    # info = @benchmark solve!($prob, a, b) setup=(a=deepcopy(x̄), b=deepcopy(ū))
    # @test info.allocs == 0
end