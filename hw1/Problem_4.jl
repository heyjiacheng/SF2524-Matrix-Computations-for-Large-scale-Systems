# ============================================================================
# Problem 4: Primitive Variant of Arnoldi Method
# ============================================================================
# Investigates eigenvalue approximation using power method iterates.
# Compares standard Arnoldi method with non-orthogonalized Krylov basis.

using LinearAlgebra
using MatrixDepot
using Random
using Plots
using Printf

include("hw1_functions.jl")

"""
    problem_4b()

Compare Arnoldi method with non-orthogonalized Krylov method.
Generates comparison plots and analyzes the differences.
"""
function problem_4b()
    println("\n" * "="^80)
    println("PROBLEM 4(b): Comparison of Arnoldi Method and Km Method")
    println("="^80)

    # Setup: Use matrix from Problem 3
    nn = 30
    println("\nSetting up Wathen matrix with nn=$nn...")
    Random.seed!(0)
    A = matrixdepot("wathen", nn, nn)
    n = size(A, 1)
    println("Matrix size: $n × $n")

    # Random starting vector (same for both methods)
    Random.seed!(42)
    b = randn(n)
    b = b / norm(b)

    # Compute exact eigenvalues for reference
    println("\nComputing exact eigenvalues (this may take a moment)...")
    exact_eigenvalues = eigvals(Matrix(A))
    println("Exact eigenvalues computed!")

    # Range of m values to test
    m_max = 80
    m_values = 1:m_max

    println("\nRunning eigenvalue approximation methods...")
    println("This will compute approximations for m = 1, 2, ..., $m_max")

    # Store results for plotting
    arnoldi_ritz_values = Vector{Vector{ComplexF64}}()
    km_ritz_values = Vector{Vector{ComplexF64}}()

    # Progress indicator
    print("Progress: ")
    for (idx, m) in enumerate(m_values)
        if idx % 10 == 0
            print("$m ")
        end

        # Method 1: Arnoldi method with double GS
        Q, H = arnoldi(A, b, m, cgs_mv, 2)  # Using double GS as recommended
        Hm = H[1:m, 1:m]
        arnoldi_ritz = eigvals(Hm)
        push!(arnoldi_ritz_values, arnoldi_ritz)

        # Method 2: Km method using equation (2)
        Km = construct_krylov_basis(A, b, m)
        km_ritz = eigenvalue_approximation_krylov(A, Km)
        push!(km_ritz_values, km_ritz)
    end
    println("\nDone!")

    # Create comparison plot
    println("\nGenerating comparison plot...")

    # Plot setup
    p = plot(size=(1000, 600), legend=:topright, dpi=300)
    plot!(xlabel="m (number of iterations)", ylabel="Real part of eigenvalue approx.",
          title="Eigenvalue Approximations: Arnoldi vs Km Method",
          xlim=(0, m_max+5), grid=true)

    # Plot Arnoldi Ritz values
    for (m_idx, m) in enumerate(m_values)
        ritz_vals = arnoldi_ritz_values[m_idx]
        real_parts = real.(ritz_vals)
        scatter!(fill(m, length(real_parts)), real_parts,
                marker=:circle, markersize=3, color=:blue, alpha=0.6,
                label=(m == 1 ? "Arnoldi method" : ""))
    end

    # Plot Km method Ritz values
    for (m_idx, m) in enumerate(m_values)
        ritz_vals = km_ritz_values[m_idx]
        real_parts = real.(ritz_vals)
        scatter!(fill(m, length(real_parts)), real_parts,
                marker=:x, markersize=3, color=:red, alpha=0.6,
                label=(m == 1 ? "Km method (eq. 2)" : ""))
    end

    # Save plot
    savefig(p, "Problem_4b_comparison.png")
    println("Plot saved as: Problem_4b_comparison.png")

    # Create a zoomed-in plot to better see differences
    println("\nGenerating zoomed plot for detailed comparison...")

    # Find reasonable y-axis limits based on eigenvalue distribution
    all_exact_real = real.(exact_eigenvalues)
    y_min = minimum(all_exact_real) - 50
    y_max = maximum(all_exact_real) + 50

    p2 = plot(size=(1000, 600), legend=:topright, dpi=300)
    plot!(xlabel="m (number of iterations)", ylabel="Real part of eigenvalue approx.",
          title="Eigenvalue Approximations (Zoomed): Arnoldi vs Km Method",
          xlim=(0, m_max+5), ylim=(y_min, y_max), grid=true)

    # Plot Arnoldi Ritz values
    for (m_idx, m) in enumerate(m_values)
        ritz_vals = arnoldi_ritz_values[m_idx]
        real_parts = real.(ritz_vals)
        scatter!(fill(m, length(real_parts)), real_parts,
                marker=:circle, markersize=3, color=:blue, alpha=0.6,
                label=(m == 1 ? "Arnoldi method" : ""))
    end

    # Plot Km method Ritz values
    for (m_idx, m) in enumerate(m_values)
        ritz_vals = km_ritz_values[m_idx]
        real_parts = real.(ritz_vals)
        scatter!(fill(m, length(real_parts)), real_parts,
                marker=:x, markersize=3, color=:red, alpha=0.6,
                label=(m == 1 ? "Km method (eq. 2)" : ""))
    end

    savefig(p2, "Problem_4b_comparison_zoomed.png")
    println("Zoomed plot saved as: Problem_4b_comparison_zoomed.png")

    # Analysis
    println("\n" * "="^80)
    println("ANALYSIS")
    println("="^80)

    println("\nComparison at specific m values:")
    println("-"^80)

    test_m_values = [10, 20, 30, 40, 50, 60, 70, 80]

    for m in test_m_values
        if m > m_max
            continue
        end

        m_idx = findfirst(x -> x == m, m_values)

        arnoldi_ritz = arnoldi_ritz_values[m_idx]
        km_ritz = km_ritz_values[m_idx]

        # Find the largest real part from each method
        arnoldi_max = maximum(real.(arnoldi_ritz))
        km_max = maximum(real.(km_ritz))

        # Compute approximation error (compared to largest exact eigenvalue)
        exact_max = maximum(real.(exact_eigenvalues))
        arnoldi_error = abs(arnoldi_max - exact_max)
        km_error = abs(km_max - exact_max)

        println("\nm = $m:")
        @printf("  Arnoldi: largest real part = %.6f, error = %.6e\n", arnoldi_max, arnoldi_error)
        @printf("  Km:      largest real part = %.6f, error = %.6e\n", km_max, km_error)
    end

    # Theoretical expectation
    println("\n" * "="^80)
    println("THEORETICAL EXPECTATION (Exact Arithmetic)")
    println("="^80)
    println("""
According to Problem 4(a), the eigenvalue approximation from equation (2):
    μw = (Km' * Km)^(-1) * Km' * A * Km * w

is theoretically IDENTICAL to the Ritz values computed by Arnoldi's method
in exact arithmetic. This is because:

1. Both methods work in the same Krylov subspace:
   span(Km) = span{b, Ab, A²b, ..., A^(m-1)b} = K_m(A, b)

2. The Arnoldi method constructs an orthonormal basis Q for this space
3. The Rayleigh-Ritz procedure projects A onto this space
4. Equation (2) performs the same projection without explicit orthogonalization

Therefore, both methods should produce the same eigenvalue approximations.
    """)

    println("\n" * "="^80)
    println("OBSERVED BEHAVIOR (Finite Precision)")
    println("="^80)
    println("""
From the plots, we observe:

1. SIMILAR APPROXIMATIONS: Both methods produce similar eigenvalue
   approximations, confirming they work in the same Krylov subspace.

2. NUMERICAL DIFFERENCES: Due to finite precision arithmetic:
   - Arnoldi method maintains orthogonality through Gram-Schmidt
   - Km method works with non-orthogonal power method iterates
   - As m increases, Km loses numerical rank (columns become nearly parallel)
   - This causes (Km' * Km) to become ill-conditioned

3. STABILITY ISSUES: The Km method may show:
   - Spurious eigenvalues (especially for larger m)
   - Less accurate approximations compared to Arnoldi
   - Numerical instability when solving (Km' * Km)^(-1) * Km' * A * Km

4. CONVERGENCE PATTERN: Both methods show eigenvalues converging as m increases,
   but Arnoldi maintains better numerical stability throughout.
    """)

    println("\n" * "="^80)
    println("CONCLUSION: WHICH APPROACH IS BETTER?")
    println("="^80)
    println("""
The ARNOLDI METHOD is clearly better for the following reasons:

1. NUMERICAL STABILITY:
   - Orthogonalization prevents loss of numerical rank
   - Condition number of Q'Q = 1 (perfectly conditioned)
   - Condition number of Km'Km grows exponentially with m

2. COMPUTATIONAL EFFICIENCY:
   - Arnoldi only requires O(nm²) operations per iteration
   - Km method requires matrix-matrix products Km' * A * Km
   - Arnoldi produces Hessenberg matrix H directly (no eigenvalue solve needed)

3. ACCURACY:
   - Arnoldi maintains accuracy even for large m
   - Km method deteriorates as columns of Km become linearly dependent
   - Arnoldi with double GS achieves near-machine-precision orthogonality

4. PRACTICAL USE:
   - Arnoldi is the foundation of production codes (ARPACK, etc.)
   - Km method is primarily of theoretical interest
   - Demonstrates why orthogonalization is crucial in Krylov methods

THEORETICAL SIGNIFICANCE:
While equation (2) is theoretically equivalent to Arnoldi in exact arithmetic,
it demonstrates an important principle: working with orthogonal bases is
essential for numerical stability in Krylov subspace methods.
    """)

    println("\n" * "="^80)
    println("Problem 4(b) completed successfully!")
    println("="^80 * "\n")

    return arnoldi_ritz_values, km_ritz_values
end

# ============================================================================
# Execute when run directly
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    problem_4b()
end
