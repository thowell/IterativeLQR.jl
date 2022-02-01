
# ## benchmarking
using BenchmarkTools
using InteractiveUtils

objective!(prob.s_data, prob.m_data, mode=:nominal)
@benchmark objective!($prob.s_data, $prob.m_data, mode=:nominal)
@code_warntype objective!(prob.s_data, prob.m_data, mode=:nominal)

derivatives!(prob.m_data, mode=:nominal)
@benchmark derivatives!($prob.m_data, mode=:nominal)
@code_warntype derivatives!(prob.m_data, mode=:nominal)

mode = :nominal
IterativeLQR.backward_pass!(prob.p_data, prob.m_data, mode=mode)
@benchmark IterativeLQR.backward_pass!($prob.p_data, $prob.m_data, mode=:nominal)# setup=(mode=:nominal)
@benchmark IterativeLQR.backward_pass!($prob.p_data, $prob.m_data, mode=:cand)# setup=(mode=:nominal)
@code_warntype IterativeLQR.backward_pass!(prob.p_data, prob.m_data, mode=mode)

lagrangian_gradient!(prob.s_data, prob.p_data, prob.m_data)
@benchmark lagrangian_gradient!($prob.s_data, $prob.p_data, $prob.m_data)
@code_warntype lagrangian_gradient!(prob.s_data, prob.p_data, prob.m_data)

IterativeLQR.Δz!(prob.m_data, prob.p_data, prob.s_data)
@benchmark IterativeLQR.Δz!($prob.m_data, $prob.p_data, $prob.s_data)
@code_warntype IterativeLQR.Δz!(prob.m_data, prob.p_data, prob.s_data)

rollout!(prob.p_data, prob.m_data, α=prob.s_data.α[1])
@benchmark rollout!($prob.p_data, $prob.m_data, α=$prob.s_data.α[1])
@code_warntype rollout!(prob.p_data, prob.m_data, α=prob.s_data.α[1])

augmented_lagrangian_update!(prob.m_data.obj)
@benchmark augmented_lagrangian_update!($prob.m_data.obj)
@code_warntype augmented_lagrangian_update!(prob.m_data.obj)

constraint_violation(prob.m_data.obj.c_data)
@benchmark constraint_violation($prob.m_data.obj.c_data)
@code_warntype constraint_violation(prob.m_data.obj.c_data)

initialize_controls!(prob, ū)
@benchmark initialize_controls!($prob, $ū)
@code_warntype initialize_controls!(prob, ū)

initialize_states!(prob, x̄)
@benchmark initialize_states!($prob, $x̄)
@code_warntype initialize_states!(prob, x̄)

function _forward_pass!(prob::Solver, x, u) 
    initialize_controls!(prob, u)
    initialize_states!(prob, x)

    forward_pass!(prob.p_data, prob.m_data, prob.s_data,
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
a = prob.m_data.obj.ρ[t]
b = prob.m_data.obj.a[t]
c = prob.m_data.obj.Iρ[t]
@benchmark $c .= $(Diagonal(a .* b))