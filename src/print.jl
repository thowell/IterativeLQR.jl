function solver_info()
    println(crayon"bold red",
    "
 ___ _                _   _         _    ___  ___
|_ _| |_ ___ _ _ __ _| |_(_)_ _____| |  / _ \\| _ \\
 | ||  _/ -_) '_/ _` |  _| \\ V / -_) |_| (_) |   /
|___|\\__\\___|_| \\__,_|\\__|_|\\_/\\___|____\\__\\_\\_|_\\
    ")
    println(crayon"reset bold black",
    "Taylor Howell and Simon Le Cleac'h")
    println("Robotic Exploration Lab")
    println("Stanford University\n")
    print(crayon"reset")
end

function iteration_status(
    total_iterations,
    outer_iterations,
    inner_iterations,
    residual_violation,
    constraint_violation,
    penalty,
    step_size,
    print_frequency,
    )

    # header
    if rem(total_iterations - 1, print_frequency) == 0
        @printf "------------------------------------------------------------------------------------------------\n"
        @printf "total  outer  inner |residual| |constraint|   penalty   step  \n"
        @printf "------------------------------------------------------------------------------------------------\n"
    end

    # iteration information
    @printf("%3d     %2d    %3d   %9.2e  %9.2e   %9.2e   %9.2e \n",
        total_iterations,
        outer_iterations,
        inner_iterations,
        residual_violation,
        equality_violation,
        penalty,
        step_size)
end
