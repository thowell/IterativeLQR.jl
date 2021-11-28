using ForwardDiff 
using Symbolics


# function 
my_func(x) = cos(x[1])

# variables
@variables x[1:1]

# symbolic functions
f = my_func(x)
df = Symbolics.gradient(f, x)

# julia functions 
f_func = eval(Symbolics.build_function(f, x))
df_func = eval(Symbolics.build_function(df, x)[1])

# eval functions
f_func([0.5 * π])
df_func([0.5 * π])

# diff gradient function (autodiff)
ForwardDiff.jacobian(df_func, [0.5 * π])

# diff gradient function (symbolics)
dfs = df_func(x)
ddfs = Symbolics.gradient(dfs[1], x)


## 
using Symbolics
using LinearAlgebra

function midpoint(f,x,u,dt)
    return x + dt*f(x + 0.5*dt*f(x,u),u)
end

function car(x, u)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

@variables x[1:3] u[1:2]
dt = .1
val = midpoint(car,x,u,dt)
@show val
jac = Symbolics.jacobian(val,[x;u])
@show jac
