# PREAMBLE

# PKG_SETUP

# ## Setup

using IterativeLQR 
using LinearAlgebra
using Plots

# ## horizon 
T = 101 

# ## acrobot 
nx = 4 
nu = 1 
nw = 0 

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

    qdd = M(q, w) \ (-1.0 * C(x, w) * v
            + τ(q, w) + B(q, w) * u[1] - [friction1; friction2] .* v)

    return [x[3]; x[4]; qdd[1]; qdd[2]]
end

function midpoint_implicit(y, x, u, w)
    h = 0.05 # timestep 
    y - (x + h * acrobot(0.5 * (x + y), u, w))
end

# ## model
dt = Dynamics(midpoint_implicit, nx, nx, nu, nw=nw)
dyn = [dt for t = 1:T-1] 
model = DynamicsModel(dyn)

# ## initialization
x1 = [0.0; 0.0; 0.0; 0.0] 
xT = [0.0; π; 0.0; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
oT = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4])
ct = Cost(ot, nx, nu, nw, [t for t = 1:T-1])
cT = Cost(oT, nx, 0, nw, [T])
obj = [ct, cT]

# ## constraints
x_init = Bound(nx, nu, [1], xl=x1, xu=x1)
x_goal = Bound(nx, 0, [T], xl=xT, xu=xT)
cons = ConstraintSet([x_init, x_goal], [StageConstraint()])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, options=Options())

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [1.0 * randn(nu) for t = 1:T-1]
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = x_interpolation[t]
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
initialize!(s, z0)

# ## solve
@time solve!(s)

# ## solution
@show trajopt.x[1]
@show trajopt.x[T]

# ## state
plot(hcat(trajopt.x...)')

# ## control
plot(hcat(trajopt.u[1:end-1]..., trajopt.u[end-1])', linetype = :steppost)