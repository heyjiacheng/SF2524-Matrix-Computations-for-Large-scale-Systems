# ============================================================================
# Problem 2: Eigenvalue Methods Comparison
# ============================================================================

using LinearAlgebra
using Plots

include("hw1_functions.jl")

"""
    problem_2()

Compare different eigenvalue iteration methods:
- Power Method
- Inverse Iteration
- Rayleigh Quotient Iteration
- Effect of matrix perturbation on convergence
"""
function problem_2()
    println("\n" * "="^80)
    println("PROBLEM 2: Eigenvalue Methods Comparison")
    println("="^80)

    # Test matrix
    B = [4 1 1; 1 3 0; 1 0 2]
    v0 = [1; 1; 1]
    tol = 1e-10

    # Get exact eigenvalues
    eigenvalues = eigvals(B)
    println("\nExact eigenvalues: ", sort(eigenvalues, rev=true))
    println("Ratio |λ₂/λ₁| = ", abs(eigenvalues[2] / eigenvalues[3]))

    # Part (a): Power Method
    println("\n" * "-"^80)
    println("Part (a): Power Method")
    println("-"^80)
    error_power = power_method(B, v0; tol=tol)

    # Part (b): Inverse Iteration with shift μ = 3
    println("\n" * "-"^80)
    println("Part (b): Inverse Iteration (shift μ = 3)")
    println("-"^80)
    error_inverse = inverse_iteration(B, v0, 3; tol=tol)

    # Part (c): Rayleigh Quotient Iteration
    println("\n" * "-"^80)
    println("Part (c): Rayleigh Quotient Iteration")
    println("-"^80)
    error_rqi = rayleigh_quotient_iteration(B, v0; tol=tol, name="Original")

    # Part (d): Perturbed matrix
    println("\n" * "-"^80)
    println("Part (d): Perturbed Matrix (B₂₁ = 1.5)")
    println("-"^80)
    B_perturbed = [4 1 1; 1.5 3 0; 1 0 2]
    println("Original symmetric: ", issymmetric(B))
    println("Perturbed symmetric: ", issymmetric(B_perturbed))
    println("Exact eigenvalues (perturbed): ", sort(eigvals(B_perturbed), rev=true))
    error_rqi_perturbed = rayleigh_quotient_iteration(B_perturbed, v0; tol=tol, name="Perturbed")

    # Combined comparison plot
    println("\n" * "-"^80)
    println("Creating comparison plot...")

    p = plot(yaxis=:log,
             xlabel="Iteration",
             ylabel="Relative Error",
             title="Eigenvalue Methods Comparison",
             legend=:topright,
             size=(800, 600))

    plot!(p, 1:length(error_power), error_power,
          label="Power Method", marker=:circle, linewidth=2)
    plot!(p, 1:length(error_inverse), error_inverse,
          label="Inverse Iteration (μ=3)", marker=:square, linewidth=2)
    plot!(p, 1:length(error_rqi), error_rqi,
          label="Rayleigh Quotient", marker=:diamond, linewidth=2)
    plot!(p, 1:length(error_rqi_perturbed), error_rqi_perturbed,
          label="Rayleigh Quotient (Perturbed)", marker=:star, linewidth=2)

    savefig(p, "Problem2_comparison.png")
    println("Saved: Problem2_comparison.png")

    println("\n" * "="^80)
    println("ANALYSIS")
    println("="^80)
    println("\n(a) Power Method: Linear convergence, rate ∝ |λ₂/λ₁|")
    println("(b) Inverse Iteration: Fast convergence to eigenvalue near shift")
    println("(c) Rayleigh Quotient: Cubic convergence (very fast!)")
    println("(d) Perturbed Matrix: Breaking symmetry affects convergence")
    println("="^80)
end

# Run the problem
if abspath(PROGRAM_FILE) == @__FILE__
    problem_2()
end
