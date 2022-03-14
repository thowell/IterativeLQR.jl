@testset "Solve: acrobot" begin 
    # ## horizon 
    T = 101 

    # ## acrobot 
    num_state = 4 
    num_action = 1 
    num_parameter = 0 

    function acrobot(x, u, w)
        mass1 = 1.0  
        inertia1 = 0.33  
        length1 = 1.0 
        lengthcom1 = 0.5 

        mass2 = 1.0  
        inertia2 = 0.33  
        length2 = 1.0 
        lengthcom2 = 0.5 

        gravity = 9.81 
        friction1 = 0.1 
        friction2 = 0.1

        function M(x, w)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

            return [a b; b c]
        end

        function Minv(x, w) 
            m = M(x, w) 
            a = m[1, 1] 
            b = m[1, 2] 
            c = m[2, 1] 
            d = m[2, 2]
            1.0 / (a * d - b * c) * [d -b;-c a]
        end

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

        function B(x, w)
            [0.0; 1.0]
        end

        q = view(x, 1:2)
        v = view(x, 3:4)

        qdd = Minv(q, w) * (-1.0 * C(x, w) * v
                + τ(q, w) + B(q, w) * u[1] - [friction1; friction2] .* v)

        return [x[3]; x[4]; qdd[1]; qdd[2]]
    end

    function midpoint_explicit(x, u, w)
        h = 0.1 # timestep 
        x + h * acrobot(x + 0.5 * h * acrobot(x, u, w), u, w)
    end

    # ## model
    dyn = Dynamics(midpoint_explicit, num_state, num_action, num_parameter=num_parameter)
    model = [dyn for t = 1:T-1] 

    # ## initialization
    x1 = [0.0; 0.0; 0.0; 0.0] 
    xT = [0.0; π; 0.0; 0.0]
    ū = [1.0e-3 * randn(num_action) for t = 1:T-1] 
    w = [zeros(num_parameter) for t = 1:T]
    x̄ = rollout(model, x1, ū, w)

    # ## objective 
    ot = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4])
    ct = Cost(ot, num_state, num_action, num_parameter=num_parameter)
    cT = Cost(oT, num_state, 0, num_parameter=num_parameter)
    obj = [[ct for t = 1:T-1]..., cT]

    # ## constraints
    goal(x, u, w) = x - xT

    cont = Constraint()
    conT = Constraint(goal, num_state, 0)
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
    s = solver(model, obj, cons, options=options)
    initialize_controls!(s, ū) 
    initialize_states!(s, x̄)

    # ## solve
    solve!(s)

    # ## solution
    x_sol, u_sol = get_trajectory(s)

    @test norm(x_sol[T] - xT, Inf) < options.constraint_tolerance

    # ## allocations
    # info = @benchmark solve!($prob, a, b) setup=(a=deepcopy(x̄), b=deepcopy(ū))
    # @test info.allocs == 0
end