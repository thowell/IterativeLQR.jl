
# ## benchmarking
using BenchmarkTools
using InteractiveUtils

cost!(prob.solver_data, prob.problem, mode=:nominal)
@benchmark cost!($prob.solver_data, $prob.problem, mode=:nominal)
@code_warntype cost!(prob.solver_data, prob.problem, mode=:nominal)

gradients!(prob.problem, mode=:nominal)
@benchmark gradients!($prob.problem, mode=:nominal)
@code_warntype gradients!(prob.problem, mode=:nominal)

mode = :nominal
IterativeLQR.backward_pass!(prob.policy, prob.problem, mode=mode)
@benchmark IterativeLQR.backward_pass!($prob.policy, $prob.problem, mode=:nominal)# setup=(mode=:nominal)
@benchmark IterativeLQR.backward_pass!($prob.policy, $prob.problem, mode=:cand)# setup=(mode=:nominal)
@code_warntype IterativeLQR.backward_pass!(prob.policy, prob.problem, mode=mode)

lagrangian_gradient!(prob.solver_data, prob.policy, prob.problem)
@benchmark lagrangian_gradient!($prob.solver_data, $prob.policy, $prob.problem)
@code_warntype lagrangian_gradient!(prob.solver_data, prob.policy, prob.problem)

IterativeLQR.Δz!(prob.problem, prob.policy, prob.solver_data)
@benchmark IterativeLQR.Δz!($prob.problem, $prob.policy, $prob.solver_data)
@code_warntype IterativeLQR.Δz!(prob.problem, prob.policy, prob.solver_data)

rollout!(prob.policy, prob.problem, α=prob.solver_data.α[1])
@benchmark rollout!($prob.policy, $prob.problem, α=$prob.solver_data.α[1])
@code_warntype rollout!(prob.policy, prob.problem, α=prob.solver_data.α[1])

augmented_lagrangian_update!(prob.problem.obj)
@benchmark augmented_lagrangian_update!($prob.problem.obj)
@code_warntype augmented_lagrangian_update!(prob.problem.obj)

constraint_violation(prob.problem.obj.constraint_data)
@benchmark constraint_violation($prob.problem.obj.constraint_data)
@code_warntype constraint_violation(prob.problem.obj.constraint_data)

initialize_controls!(prob, ū)
@benchmark initialize_controls!($prob, $ū)
@code_warntype initialize_controls!(prob, ū)

initialize_states!(prob, x̄)
@benchmark initialize_states!($prob, $x̄)
@code_warntype initialize_states!(prob, x̄)

function _forward_pass!(prob::Solver, x, u) 
    initialize_controls!(prob, u)
    initialize_states!(prob, x)

    forward_pass!(prob.policy, prob.problem, prob.solver_data,
        α_min = 1.0e-5,
        linesearch = :armijo)
end

_forward_pass!(prob, x̄, ū)
@benchmark _forward_pass!($prob, $x̄, $ū)
@code_warntype _forward_pass!(prob, x̄, ū)


A = Diagonal(ones(5))
a = rand(5)
b = ones(Int, 5)
using BenchmarkTools
@benchmark $A .= $(Diagonal(b .* a))

t = 1
a = prob.problem.obj.ρ[t]
b = prob.problem.obj.a[t]
c = prob.problem.obj.Iρ[t]
@benchmark $c .= $(Diagonal(a .* b))