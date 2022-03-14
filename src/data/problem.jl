

mutable struct ProblemData{T,X,U,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX}
    # current trajectory
    states::Vector{X}
    actions::Vector{U}

    # disturbance trajectory
    parameters::Vector{D}

    # nominal trajectory
    nominal_states::Vector{X}
    nominal_actions::Vector{U}

    # model data
    model::ModelData{T,FX,FU,FW}

    # objective data
    objective::ObjectiveData{O,OX,OU,OXX,OUU,OUX}

    # trajectory: z = (x1,..., xT, u1,..., uT-1) | Δz = (Δx1..., ΔxT, Δu1,..., ΔuT-1)
    trajectory::Vector{T}
end

function problem_data(dynamics, costs; 
    parameters=[[zeros(d.num_parameter) for d in dynamics]..., zeros(0)])

    length(parameters) == length(dynamics) && (parameters = [parameters..., zeros(0)])
    @assert length(dynamics) + 1 == length(parameters)
    @assert length(dynamics) + 1 == length(costs)

	states = [[zeros(d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state)]
    actions = [[zeros(d.num_action) for d in dynamics]..., zeros(0)]

    nominal_states = [[zeros(d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state)]
    nominal_actions = [[zeros(d.num_action) for d in dynamics]..., zeros(0)]

    model = model_data(dynamics)
    objective = objective_data(dynamics, costs)

    trajectory = zeros(num_trajectory(dynamics))

    ProblemData(states, actions, parameters, nominal_states, nominal_actions, model, objective, trajectory)
end