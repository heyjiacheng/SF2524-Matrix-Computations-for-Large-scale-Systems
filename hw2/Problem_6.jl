# Problem 6: Heatsink timing experiments
# (a) Matrix-vector product timing
# (c) Backslash timing
# (d) Preconditioner accuracy and timing comparison

using LinearAlgebra
using MAT
using SparseArrays
using Statistics

"""
Load heatsink data and build the reduced system matrix A.
"""
function build_heatsink_matrix(model_num::Int)
    filename = "hw2/heatsink_exercise_ref$(model_num).mat"
    data = matread(filename)

    K = data["K"]
    coords = data["coords"]
    baseNodes = convert.(Int64, data["baseNodes"])

    N = size(coords, 1)

    # Build free/fixed node indices (same as heatsink_build_matrices.jl)
    fixed = falses(N)
    fixed[baseNodes] .= true
    free = .!fixed

    # Reduced system matrix
    A = K[free, free]

    return A, N, sum(free)
end

"""
Time matrix-vector product for a given matrix.
Runs num_trials times and returns individual times and average.
"""
function time_matvec(A, num_trials::Int=10)
    n = size(A, 1)
    x = randn(n)  # Random vector

    # Warm-up run (JIT compilation)
    _ = A * x

    times = Float64[]

    for i = 1:num_trials
        x = randn(n)  # Use fresh random vector each time
        t_start = time_ns()
        _ = A * x
        t_end = time_ns()
        push!(times, (t_end - t_start) / 1e9)  # Convert to seconds
    end

    return times, mean(times), std(times)
end

"""
Time backslash (A\\b) for a given matrix.
Runs num_trials times and returns individual times and average.
"""
function time_backslash(A, num_trials::Int=10)
    n = size(A, 1)
    b = randn(n)  # Random RHS vector

    # Warm-up run (JIT compilation)
    _ = A \ b

    times = Float64[]

    for i = 1:num_trials
        b = randn(n)  # Use fresh random vector each time
        t_start = time_ns()
        _ = A \ b
        t_end = time_ns()
        push!(times, (t_end - t_start) / 1e9)  # Convert to seconds
    end

    return times, mean(times), std(times)
end

"""
Load preconditioner T1, T2 from heatsink_precond_refX.mat
"""
function load_preconditioner(model_num::Int)
    filename = "hw2/heatsink_precond_ref$(model_num).mat"
    data = matread(filename)
    T1 = data["T1"]
    T2 = data["T2"]
    return T1, T2
end

"""
Compute preconditioner accuracy: norm(M^{-1}(A*z) - z) / norm(A*z)
For M^{-1} = T2^{-1} * T1^{-1}, this is: norm(T2\\(T1\\(A*z)) - z) / norm(A*z)
"""
function precond_accuracy_M(A, T1, T2, z)
    Az = A * z
    Minv_Az = T2 \ (T1 \ Az)
    return norm(Minv_Az - z) / norm(Az)
end

"""
Compute diagonal preconditioner accuracy: norm(D^{-1}(A*z) - z) / norm(A*z)
Where D = diag(A)
"""
function precond_accuracy_D(A, D, z)
    Az = A * z
    Dinv_Az = D \ Az
    return norm(Dinv_Az - z) / norm(Az)
end

"""
Time preconditioner M^{-1} = T2^{-1} * T1^{-1} application.
"""
function time_precond_M(A, T1, T2, num_trials::Int=10)
    n = size(A, 1)
    z = randn(n)
    Az = A * z

    # Warm-up
    _ = T2 \ (T1 \ Az)

    times = Float64[]

    for i = 1:num_trials
        z = randn(n)
        Az = A * z
        t_start = time_ns()
        _ = T2 \ (T1 \ Az)
        t_end = time_ns()
        push!(times, (t_end - t_start) / 1e9)
    end

    return times, mean(times), std(times)
end

"""
Time diagonal preconditioner D^{-1} application.
"""
function time_precond_D(A, D, num_trials::Int=10)
    n = size(A, 1)
    z = randn(n)
    Az = A * z

    # Warm-up
    _ = D \ Az

    times = Float64[]

    for i = 1:num_trials
        z = randn(n)
        Az = A * z
        t_start = time_ns()
        _ = D \ Az
        t_end = time_ns()
        push!(times, (t_end - t_start) / 1e9)
    end

    return times, mean(times), std(times)
end

"""
Main function for Problem 6(a)
"""
function main()
    println("=" ^ 60)
    println("Problem 6(a): Matrix-Vector Product Timing")
    println("=" ^ 60)

    num_trials = 10

    for X in [1, 2]
        println("\n--- Model X = $X ---")

        # Build matrix
        A, N_total, n_free = build_heatsink_matrix(X)

        println("Total nodes: $N_total")
        println("Free DOFs (matrix size): $n_free × $n_free")
        println("Number of nonzeros: $(nnz(A))")

        # Time matrix-vector product
        times, avg_time, std_time = time_matvec(A, num_trials)

        println("\nMatrix-vector product timing ($num_trials trials):")
        for (i, t) in enumerate(times)
            println("  Trial $i: $(round(t * 1e6, digits=3)) μs")
        end

        println("\nAverage time: $(round(avg_time * 1e6, digits=3)) μs")
        println("Std deviation: $(round(std_time * 1e6, digits=3)) μs")
        println("Average time: $(round(avg_time * 1e3, digits=6)) ms")
    end

    println("\n" * "=" ^ 60)
end

"""
Main function for Problem 6(c): Backslash timing
"""
function main_6c()
    println("=" ^ 60)
    println("Problem 6(c): Backslash (A\\\\b) Timing")
    println("=" ^ 60)

    num_trials = 10

    for X in [1, 2]
        println("\n--- Model X = $X ---")

        # Build matrix
        A, N_total, n_free = build_heatsink_matrix(X)

        println("Total nodes: $N_total")
        println("Free DOFs (matrix size): $n_free × $n_free")
        println("Number of nonzeros: $(nnz(A))")

        # Time backslash
        times, avg_time, std_time = time_backslash(A, num_trials)

        println("\nBackslash timing ($num_trials trials):")
        for (i, t) in enumerate(times)
            println("  Trial $i: $(round(t * 1e3, digits=3)) ms")
        end

        println("\nAverage time: $(round(avg_time * 1e3, digits=3)) ms")
        println("Std deviation: $(round(std_time * 1e3, digits=3)) ms")
        println("Average time: $(round(avg_time, digits=6)) s")
    end

    println("\n" * "=" ^ 60)
end

"""
Main function for Problem 6(d): Preconditioner comparison
"""
function main_6d()
    println("=" ^ 60)
    println("Problem 6(d): Preconditioner Accuracy and Timing")
    println("=" ^ 60)

    num_trials = 10

    for X in [1, 2]
        println("\n--- Model X = $X ---")

        # Build matrix A
        A, N_total, n_free = build_heatsink_matrix(X)

        # Load preconditioner
        T1, T2 = load_preconditioner(X)

        # Build diagonal preconditioner D = diag(A)
        D = spdiagm(0 => diag(A))

        println("Matrix size: $n_free × $n_free")

        # --- Accuracy comparison ---
        println("\n[Accuracy] (10 random z vectors):")
        println("  norm(T2\\(T1\\(A*z))-z)/norm(A*z)  vs  norm(D\\(A*z)-z)/norm(A*z)")

        acc_M = Float64[]
        acc_D = Float64[]

        for i = 1:num_trials
            z = randn(n_free)
            push!(acc_M, precond_accuracy_M(A, T1, T2, z))
            push!(acc_D, precond_accuracy_D(A, D, z))
        end

        println("\n  Trial |    M precond    |    D precond")
        println("  ------|-----------------|----------------")
        for i = 1:num_trials
            println("    $(lpad(i, 2))  |  $(round(acc_M[i], sigdigits=6))  |  $(round(acc_D[i], sigdigits=6))")
        end

        println("\n  Average accuracy (M): $(round(mean(acc_M), sigdigits=6))")
        println("  Average accuracy (D): $(round(mean(acc_D), sigdigits=6))")

        if mean(acc_M) < mean(acc_D)
            println("  → M^{-1} = T2^{-1}T1^{-1} is MORE accurate")
        else
            println("  → D^{-1} (diagonal) is MORE accurate")
        end

        # --- Timing comparison ---
        println("\n[Timing] Preconditioner application ($num_trials trials):")

        times_M, avg_M, std_M = time_precond_M(A, T1, T2, num_trials)
        times_D, avg_D, std_D = time_precond_D(A, D, num_trials)

        println("  M precond average time: $(round(avg_M * 1e6, digits=3)) μs")
        println("  D precond average time: $(round(avg_D * 1e6, digits=3)) μs")

        if avg_M < avg_D
            println("  → M preconditioner is FASTER")
        else
            println("  → D (diagonal) preconditioner is FASTER")
        end
    end

    println("\n" * "=" ^ 60)
end

# Run all parts
main()      # Problem 6(a)
main_6c()   # Problem 6(c)
main_6d()   # Problem 6(d)
