# ============================================================================
# Problem 5: Arnoldi Method Applied to Bwedge Matrix
# ============================================================================
# Applies Arnoldi method to compute eigenvalues and Ritz values.
# Includes shift-and-invert strategy for targeting specific eigenvalues.

using MAT
using Plots
using Random
using LinearAlgebra

include("hw1_functions.jl")

# ============================================================================
# Problem 5: Standard Arnoldi Analysis
# ============================================================================

"""
    problem_5()

Apply Arnoldi method to Bwedge matrix with varying iteration counts.
Generates plots comparing exact eigenvalues with Ritz values.
"""
function problem_5()
    println("\n" * "="^80)
    println("PROBLEM 5: Arnoldi Method on Bwedge Matrix")
    println("="^80)

    Random.seed!(0)
    B = matread("Bwedge.mat")["B"]
    eigenvalues = eigvals(B)

    # Plot exact eigenvalues
    p = plot(real(eigenvalues), imag(eigenvalues),
             seriestype=:scatter,
             xlabel="Real Part",
             ylabel="Imaginary Part",
             title="Eigenvalues of B",
             legend=false)
    savefig(p, "eigenvalues_plot.png")
    println("Saved: eigenvalues_plot.png")

    # Test different iteration counts
    m_values = [2, 4, 8, 10, 20, 30, 40]

    for m in m_values
        println("\n" * "-"^80)
        println("m = $m")

        Q, H = arnoldi(B, randn(size(B, 1)), m, classical_gram_schmidt, 2)
        H_sub = H[1:m, 1:m]
        ritz_values = eigvals(H_sub)

        # Plot eigenvalues and Ritz values
        p = plot(real(eigenvalues), imag(eigenvalues),
                 seriestype=:scatter,
                 xlabel="Real Part",
                 ylabel="Imaginary Part",
                 title="Eigenvalues and Ritz Values (m=$m)",
                 label="Exact Eigenvalues")

        plot!(p, real(ritz_values), imag(ritz_values),
              seriestype=:scatter,
              label="Ritz Values (m=$m)")

        # Compute errors
        errors = compare_eigenvalues_ritzvalues(eigenvalues, ritz_values)
        error_values = [e[2] for e in errors]
        min_idx = argmin(error_values)
        min_error = errors[min_idx]

        println("  Best approximated eigenvalue: ", min_error[1])
        println("  Minimum relative error: ", min_error[2])

        savefig(p, "Ritz_values_plot_m_$(m).png")
    end

    println("\n" * "="^80)
end

# ============================================================================
# Problem 5c: Shift-and-Invert Strategy
# ============================================================================

"""
    problem_5c_shift_invert()

Use shift-and-invert Arnoldi to target specific eigenvalues.
Tests multiple shifts to approximate eigenvalues near the shift.
"""
function problem_5c_shift_invert()
    println("\n" * "="^80)
    println("PROBLEM 5c: Shift-and-Invert Arnoldi")
    println("="^80)

    Random.seed!(0)
    B = matread("Bwedge.mat")["B"]
    eigenvalues = eigvals(B)

    shifts = [-10, -7 + 2im, -9.8 + 1.5im]
    m_values = [10, 20, 30]

    # Find target eigenvalue (closest to last shift)
    target_errors = compare_eigenvalues_ritzvalues(eigenvalues, [shifts[3]])
    target_eigenvalue = target_errors[1][1]
    println("Target eigenvalue: $target_eigenvalue\n")

    for shift in shifts
        println("-"^80)
        println("Shift σ = $shift")

        for m in m_values
            # Shift-and-invert: work with (B - σI)^(-1)
            A_shifted = B - shift * I
            Q, H = arnoldi(A_shifted, randn(size(A_shifted, 1)), m, classical_gram_schmidt, 2)

            # Compute Ritz values and transform back
            H_sub = H[1:m, 1:m]
            ritz_values_inv = eigvals(H_sub)
            ritz_values = shift .+ (1 ./ ritz_values_inv)

            # Compare with target eigenvalue
            errors = compare_eigenvalues_ritzvalues([target_eigenvalue], ritz_values)
            error_values = [e[2] for e in errors]
            min_idx = argmin(error_values)

            println("  m = $m:")
            println("    Best approximation: ", ritz_values[min_idx])
            println("    Relative error: ", error_values[min_idx])
        end
        println()
    end

    println("="^80)
end

# ============================================================================
# Execute when run directly
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    problem_5()
    problem_5c_shift_invert()
end
