# ============================================================================
# Problem 3: Gram-Schmidt Orthogonalization Performance in Arnoldi Method
# ============================================================================
# This file implements and compares different Gram-Schmidt variants:
# - Classical GS (matrix-vector and for-loop versions)
# - Modified GS
# - Multiple re-orthogonalization strategies

using LinearAlgebra
using MatrixDepot
using Random
using Printf
using Statistics
using Latexify

include("hw1_functions.jl")

"""
Arnoldi method with specified Gram-Schmidt orthogonalization
A: matrix
b: starting vector
m: number of iterations
GS_func: Gram-Schmidt function to use
cycles: number of re-orthogonalization cycles (default=1)
"""
function arnoldi_problem3(A, b, m, GS_func, cycles=1)
    n = length(b)
    Q = zeros(ComplexF64, n, m+1)
    H = zeros(ComplexF64, m+1, m)
    Q[:, 1] = b / norm(b)

    for k = 1:m
        w = A * Q[:, k]
        h, β, z = GS_func(Q, w, k, cycles)
        H[1:(k+1), k] = [h; β]
        if β < eps(Float64) * norm(A)
            # Breakdown - early termination
            return Q[:, 1:k], H[1:k, 1:k]
        end
        Q[:, k+1] = z / β
    end

    return Q, H
end

"""
    benchmark_gs_problem3(A, b, m, gs_func, cycles=1; n_trials=3)

Benchmark a specific Gram-Schmidt configuration for Problem 3.

# Returns
- `time`: Minimum execution time over n_trials
- `orth`: Orthogonality measure ||Q'Q - I||
"""
function benchmark_gs_problem3(A, b, m, gs_func, cycles=1; n_trials=3)
    # Warm-up run
    Q, H = arnoldi_problem3(A, b, m, gs_func, cycles)

    # Multiple timed runs for better accuracy
    times = zeros(n_trials)
    for i in 1:n_trials
        times[i] = @elapsed Q, H = arnoldi_problem3(A, b, m, gs_func, cycles)
    end

    # Final run to measure orthogonality
    Q, H = arnoldi_problem3(A, b, m, gs_func, cycles)
    Qm = Q[:, 1:m]
    orth = norm(Qm' * Qm - I)

    return minimum(times), orth
end

"""
Main function for Problem 3
"""
function Problem_3()
    println("\n" * "="^80)
    println("PROBLEM 3: Gram-Schmidt Orthogonalization Performance")
    println("="^80)

    # Matrix setup
    # Select nn such that m=100 takes approximately 1 minute
    # Starting with nn=10, can adjust based on performance
    nn = 50  # Start small, increase if needed

    println("\nSetting up matrix...")
    Random.seed!(0)
    A = matrixdepot("wathen", nn, nn)
    n = size(A, 1)
    println("Matrix size: $n × $n")

    # Starting vector
    Random.seed!(42)
    b = randn(n)
    b = b / norm(b)

    # Iteration counts
    m_values = [5, 10, 20, 50, 100]

    println("\n" * "-"^80)
    println("TABLE 1: Comparison of Different GS Implementations")
    println("-"^80)
    println("\nComparing:")
    println("  - Single GS (MV version): Classical GS with matrix-vector products")
    println("  - Single GS (for version): Classical GS with for-loops")
    println("  - Modified GS (for version): Modified GS with for-loops")
    println("\nRunning benchmarks...")

    # Table 1: Different implementations with single GS
    results1 = []
    for m in m_values
        print("  m = $m: ")

        # Single GS (MV version)
        time_mv, orth_mv = benchmark_gs_problem3(A, b, m, CGS_MV, 1)
        print("MV✓ ")

        # Single GS (for version)
        time_for, orth_for = benchmark_gs_problem3(A, b, m, CGS_for, 1)
        print("For✓ ")

        # Modified GS (for version)
        time_mgs, orth_mgs = benchmark_gs_problem3(A, b, m, MGS_for, 1)
        println("MGS✓")

        push!(results1, (m, time_mv, orth_mv, time_for, orth_for, time_mgs, orth_mgs))
    end

    # Generate LaTeX Table 1
    println("\n\nTABLE 1 - LaTeX Format:")
    println("="^80)
    # Format results with 2 decimal places
    formatted_results1 = []
    for (m, t1, o1, t2, o2, t3, o3) in results1
        push!(formatted_results1, [
            string(m),
            @sprintf("%.2f", t1),
            @sprintf("%.2e", o1),
            @sprintf("%.2f", t2),
            @sprintf("%.2e", o2),
            @sprintf("%.2f", t3),
            @sprintf("%.2e", o3)
        ])
    end
    # Create matrix with all data
    n_rows = length(formatted_results1)
    table1_data = Matrix{String}(undef, n_rows + 1, 7)
    table1_data[1, :] = ["m", "Time (MV)", "Orth (MV)", "Time (for)", "Orth (for)", "Time (MGS)", "Orth (MGS)"]
    for i in 1:n_rows
        table1_data[i+1, :] = formatted_results1[i]
    end
    latex_table1 = latexify(table1_data, env=:tabular, latex=false)
    println(latex_table1)
    println("="^80)

    println("\n" * "-"^80)
    println("TABLE 2: Multiple Re-orthogonalization (MV version only)")
    println("-"^80)
    println("\nComparing single, double, and triple Gram-Schmidt")
    println("Running benchmarks...")

    # Table 2: Multiple re-orthogonalization with MV version
    results2 = []
    for m in m_values
        print("  m = $m: ")

        # Single GS
        time_1, orth_1 = benchmark_gs_problem3(A, b, m, CGS_MV, 1)
        print("1× ")

        # Double GS
        time_2, orth_2 = benchmark_gs_problem3(A, b, m, CGS_MV, 2)
        print("2× ")

        # Triple GS
        time_3, orth_3 = benchmark_gs_problem3(A, b, m, CGS_MV, 3)
        println("3×✓")

        push!(results2, (m, time_1, orth_1, time_2, orth_2, time_3, orth_3))
    end

    # Generate LaTeX Table 2
    println("\n\nTABLE 2 - LaTeX Format:")
    println("="^80)
    # Format results with 2 decimal places
    formatted_results2 = []
    for (m, t1, o1, t2, o2, t3, o3) in results2
        push!(formatted_results2, [
            string(m),
            @sprintf("%.2f", t1),
            @sprintf("%.2e", o1),
            @sprintf("%.2f", t2),
            @sprintf("%.2e", o2),
            @sprintf("%.2f", t3),
            @sprintf("%.2e", o3)
        ])
    end
    # Create matrix with all data
    n_rows = length(formatted_results2)
    table2_data = Matrix{String}(undef, n_rows + 1, 7)
    table2_data[1, :] = ["m", "Time (1×)", "Orth (1×)", "Time (2×)", "Orth (2×)", "Time (3×)", "Orth (3×)"]
    for i in 1:n_rows
        table2_data[i+1, :] = formatted_results2[i]
    end
    latex_table2 = latexify(table2_data, env=:tabular, latex=false)
    println(latex_table2)
    println("="^80)

    # Analysis and interpretation
    println("\n" * "="^80)
    println("ANALYSIS AND INTERPRETATION")
    println("="^80)

    println("\n1. PERFORMANCE COMPARISON (Table 1):")
    println("   " * "-"^76)

    println("\n   Single GS (MV version) vs Single GS (for version):")
    println("   - The MV version uses matrix-vector products (Q' * z and Q * h)")
    println("   - This leverages BLAS Level 2 operations, which are highly optimized")
    println("   - The for-loop version uses dot products and axpy operations separately")
    println("   - Expected: MV version should be faster due to better cache locality")

    println("\n   Modified GS vs Classical GS:")
    println("   - Modified GS updates the vector z after each projection (sequential)")
    println("   - Classical GS computes all projections first, then updates (parallel)")
    println("   - Modified GS is numerically more stable in finite precision")
    println("   - Classical GS (for version) may be slightly faster but less accurate")

    println("\n   Orthogonality Quality:")
    avg_orth_mv = mean([r[3] for r in results1])
    avg_orth_for = mean([r[5] for r in results1])
    avg_orth_mgs = mean([r[7] for r in results1])
    @printf("   - Average orth (CGS MV):   %.4e\n", avg_orth_mv)
    @printf("   - Average orth (CGS for):  %.4e\n", avg_orth_for)
    @printf("   - Average orth (MGS):      %.4e\n", avg_orth_mgs)
    println("   - Modified GS typically achieves better orthogonality")
    println("   - Classical GS may lose orthogonality due to rounding errors")

    println("\n2. RE-ORTHOGONALIZATION EFFECT (Table 2):")
    println("   " * "-"^76)

    println("\n   Single vs Double vs Triple GS:")
    println("   - Re-orthogonalization applies the same GS procedure multiple times")
    println("   - Each cycle improves orthogonality at the cost of more computation")
    println("   - Double GS is often sufficient to achieve near-machine-precision orthogonality")
    println("   - Triple GS provides marginal improvement over double GS")

    println("\n   Computational Cost:")
    avg_time_1 = mean([r[2] for r in results2])
    avg_time_2 = mean([r[4] for r in results2])
    avg_time_3 = mean([r[6] for r in results2])
    @printf("   - Average time (Single GS): %.6f s\n", avg_time_1)
    @printf("   - Average time (Double GS): %.6f s (%.2f× single)\n", avg_time_2, avg_time_2/avg_time_1)
    @printf("   - Average time (Triple GS): %.6f s (%.2f× single)\n", avg_time_3, avg_time_3/avg_time_1)
    println("   - Time roughly scales linearly with number of cycles")

    println("\n   Orthogonality Improvement:")
    @printf("   - Single GS orth: %.4e\n", avg_orth_1 = mean([r[3] for r in results2]))
    @printf("   - Double GS orth: %.4e (%.2f× improvement)\n", avg_orth_2 = mean([r[5] for r in results2]), avg_orth_1/avg_orth_2)
    @printf("   - Triple GS orth: %.4e (%.2f× improvement over double)\n", avg_orth_3 = mean([r[7] for r in results2]), avg_orth_2/avg_orth_3)

    println("\n3. WHICH VERSION IS \"BEST\"?")
    println("   " * "-"^76)

    println("\n   The answer depends on the application requirements:")

    println("\n   Best for SPEED:")
    println("   - Single Classical GS (MV version)")
    println("   - Fastest execution due to optimized matrix-vector operations")
    println("   - Suitable when orthogonality errors are acceptable")

    println("\n   Best for ACCURACY:")
    println("   - Modified GS or Double Classical GS (MV version)")
    println("   - Modified GS: More stable, good orthogonality")
    println("   - Double CGS: Excellent orthogonality with manageable cost")

    println("\n   Best BALANCE:")
    println("   - Double Classical GS (MV version)")
    println("   - Combines speed of MV operations with improved orthogonality")
    println("   - Widely used in practice (e.g., ARPACK uses double GS)")

    println("\n4. FAIRNESS OF COMPARISON:")
    println("   " * "-"^76)

    println("\n   Considerations:")
    println("   - MV version benefits from hardware optimization (BLAS libraries)")
    println("   - For-loop versions may not be optimally implemented in Julia")
    println("   - In production code, for-loops could be optimized with @inbounds, @simd")
    println("   - The comparison is fair for \"as-written\" code but not for")
    println("     theoretical computational complexity")
    println("   - Modified GS is fundamentally different algorithm (sequential)")
    println("     compared to Classical GS (parallel-friendly)")

    println("\n5. PRACTICAL RECOMMENDATION:")
    println("   " * "-"^76)

    println("\n   For the Arnoldi method in practice:")
    println("   - Use Classical GS (MV version) with double orthogonalization")
    println("   - This is the approach used in high-quality libraries like ARPACK")
    println("   - Provides good balance between speed and numerical stability")
    println("   - Can handle large-scale problems efficiently")

    println("\n" * "="^80)
    println("Note: To adjust matrix size for ~1 minute runtime at m=100,")
    println("      increase nn parameter (currently nn=$nn).")
    println("      Suggested values: nn=20-50 depending on hardware.")
    println("="^80 * "\n")

    return results1, results2
end

# Run the problem
if abspath(PROGRAM_FILE) == @__FILE__
    Problem_3()
end
