# PREAMBLE

# PKG_SETUP

# ## Setup

using IterativeLQR 
using LinearAlgebra
using Plots

# ## horizon 
T = 51 

# ## acrobot 
num_state = 4 
num_action = 1 

function acrobot_continuous(x, u)
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

    function M(x)
        a = (inertia1 + inertia2 + mass2 * length1 * length1
            + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

        b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

        c = inertia2

       return [a b; b c]
    end

    function Minv(x) 
        m = M(x) 
        a = m[1, 1] 
        b = m[1, 2] 
        c = m[2, 1] 
        d = m[2, 2]
        1.0 / (a * d - b * c) * [d -b;-c a]
    end

    function τ(x)
        a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
            - mass2 * gravity * (length1 * sin(x[1])
            + lengthcom2 * sin(x[1] + x[2])))

        b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

        return [a; b]
    end

    function C(x)
        a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
        b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
        c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
        d = 0.0

        return [a b; c d]
    end

    function B(x)
        [0.0; 1.0]
    end

    q = view(x, 1:2)
    v = view(x, 3:4)

    qdd = Minv(q) * (-1.0 * C(x) * v
            + τ(q) + B(q) * u[1] - [friction1; friction2] .* v)

    return [x[3]; x[4]; qdd[1]; qdd[2]]
end

function acrobot_discrete(x, u)
    h = 0.1 # timestep 
    x + h * acrobot_continuous(x + 0.5 * h * acrobot_continuous(x, u), u)
end

# ## model
acrobot = Dynamics(acrobot_discrete, num_state, num_action)
dynamics = [acrobot for t = 1:T-1] ## best to instantiate acrobot once to reduce codegen overhead

# ## initialization
x1 = [0.0; 0.0; 0.0; 0.0] 
xT = [π; 0.0; 0.0; 0.0]
ū = [1.0 * randn(num_action) for t = 1:T-1] 
x̄ = rollout(dynamics, x1, ū)

# ## objective 
objective = [
    [Cost((x, u) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u), num_state, num_action) for t = 1:T-1]...,
    Cost((x, u) -> 0.1 * dot(x[3:4], x[3:4]), num_state, 0),
] 

# ## constraints
constraints = [
    [Constraint() for t = 1:T-1]..., 
    Constraint((x, u) -> x - xT, num_state, 0)
] 

# ## solver
solver = Solver(dynamics, objective, constraints)
initialize_controls!(solver, ū)
initialize_states!(solver, x̄)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

# ## visuals
plot(hcat(x_sol...)')
plot(hcat(u_sol...)', linetype=:steppost)

# ## benchmark allocations + timing
# info = @benchmark solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))

