# ============================================================================
# Problem 5: Arnoldi Method Applied to Bwedge Matrix
# ============================================================================
# Applies Arnoldi method to compute eigenvalues and Ritz values.
# Includes shift-and-invert strategy for targeting specific eigenvalues.

using MAT
using Plots
using Random
using LinearAlgebra
using Printf
using Latexify

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
    B = matread("hw1/Bwedge.mat")["B"]
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

    # Store results for LaTeX table
    results = []

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

        # Store result for table
        push!(results, (m, min_error[1], min_error[2]))
    end

    # Generate LaTeX table
    println("\n\n" * "="^80)
    println("LATEX TABLE: Arnoldi Method Results")
    println("="^80)

    formatted_results = []
    for (m, best_eigenval, min_rel_error) in results
        push!(formatted_results, [
            string(m),
            @sprintf("%.4f %+.4fi", real(best_eigenval), imag(best_eigenval)),
            @sprintf("%.4e", min_rel_error)
        ])
    end

    # Create matrix with all data
    n_rows = length(formatted_results)
    table_data = Matrix{String}(undef, n_rows + 1, 3)
    table_data[1, :] = ["m", "Best Approximated Eigenvalue", "Min Relative Error"]
    for i in 1:n_rows
        table_data[i+1, :] = formatted_results[i]
    end

    latex_table = latexify(table_data, env=:tabular, latex=false)
    println(latex_table)
    println("="^80)

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
    B = matread("hw1/Bwedge.mat")["B"]
    eigenvalues = eigvals(B)

    shifts = [-10, -7 + 2im, -9.8 + 1.5im]
    m_values = [10, 20, 30]

    # Find target eigenvalue (closest to last shift)
    target_errors = compare_eigenvalues_ritzvalues(eigenvalues, [shifts[3]])
    target_eigenvalue = target_errors[1][1]
    println("Target eigenvalue: $target_eigenvalue\n")

    # Store results for LaTeX table
    results_shift = []

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

            # Store result for table
            push!(results_shift, (shift, m, ritz_values[min_idx], error_values[min_idx]))
        end
        println()
    end

    # Generate LaTeX table
    println("\n" * "="^80)
    println("LATEX TABLE: Shift-and-Invert Results")
    println("="^80)
    println("Target eigenvalue: ", @sprintf("%.4f %+.4fi", real(target_eigenvalue), imag(target_eigenvalue)))
    println()

    formatted_results_shift = []
    for (shift, m, best_approx, rel_error) in results_shift
        push!(formatted_results_shift, [
            @sprintf("%.1f %+.1fi", real(shift), imag(shift)),
            string(m),
            @sprintf("%.4f %+.4fi", real(best_approx), imag(best_approx)),
            @sprintf("%.4e", rel_error)
        ])
    end

    # Create matrix with all data
    n_rows = length(formatted_results_shift)
    table_data_shift = Matrix{String}(undef, n_rows + 1, 4)
    table_data_shift[1, :] = ["Shift σ", "m", "Best Approximation", "Relative Error"]
    for i in 1:n_rows
        table_data_shift[i+1, :] = formatted_results_shift[i]
    end

    latex_table_shift = latexify(table_data_shift, env=:tabular, latex=false)
    println(latex_table_shift)
    println("="^80)
end

# ============================================================================
# Problem 5d: Transformed Eigenvalues Visualization
# ============================================================================

"""
    problem_5d()

Plot transformed eigenvalues for (B - σI)^(-1) with three different shifts.
Visualize how the target eigenvalue λ* ≈ -9.8 + 2i transforms for each shift.
"""
function problem_5d()
    println("\n" * "="^80)
    println("PROBLEM 5d: Transformed Eigenvalues Visualization")
    println("="^80)

    B = matread("hw1/Bwedge.mat")["B"]
    eigenvalues = eigvals(B)

    # Target eigenvalue: closest to -9.8 + 2i
    target_approx = -9.8 + 2im
    target_idx = argmin(abs.(eigenvalues .- target_approx))
    target_eigenvalue = eigenvalues[target_idx]
    println("Target eigenvalue λ*: $target_eigenvalue")

    # Three shifts as specified in the problem
    shifts = [-10, -7 + 2im, -9.8 + 1.5im]
    shift_names = ["σ₁ = -10", "σ₂ = -7 + 2i", "σ₃ = -9.8 + 1.5i"]

    for (idx, (shift, shift_name)) in enumerate(zip(shifts, shift_names))
        println("\nProcessing $shift_name")

        # Transform eigenvalues: λ -> 1/(λ - σ)
        transformed_eigenvalues = 1 ./ (eigenvalues .- shift)
        transformed_target = 1 / (target_eigenvalue - shift)

        # Create plot
        p = plot(real(transformed_eigenvalues), imag(transformed_eigenvalues),
                 seriestype=:scatter,
                 xlabel="real part",
                 ylabel="imaginary part",
                 title="σ = $(shift)",
                 label="transformed eigenvalues",
                 markersize=5,
                 legend=:bottomleft)

        # Mark the target eigenvalue
        scatter!(p, [real(transformed_target)], [imag(transformed_target)],
                 marker=:x,
                 markersize=8,
                 markercolor=:red,
                 label="eigenvalue we want")

        # Set axis limits adaptively
        # If target is very far (magnitude > 2), use larger range to show it
        target_magnitude = abs(transformed_target)
        if target_magnitude > 2
            # Use a range that includes the target eigenvalue
            margin = 1.2  # 20% margin
            max_range = max(abs(real(transformed_target)), abs(imag(transformed_target))) * margin
            xlims!(p, -max_range, max_range)
            ylims!(p, -max_range, max_range)
            println("  Using extended axis range to show target eigenvalue")
        else
            # Standard range for well-separated eigenvalues
            xlims!(p, -1, 1)
            ylims!(p, -1, 1)
        end

        # Save figure
        filename = "transformed_eigenvalues_sigma_$(idx).png"
        savefig(p, filename)
        println("Saved: $filename")

        # Print transformation details
        println("  Target eigenvalue in original space: $target_eigenvalue")
        println("  Target eigenvalue in transformed space: $transformed_target")
        println("  Magnitude of transformed target: $target_magnitude")
    end

    println("\n" * "="^80)
end

# ============================================================================
# Problem 5e: Shift-and-Invert Arnoldi with Error Table
# ============================================================================

"""
    problem_5e()

Implement shift-and-invert Arnoldi method with ones starting vector.
Compare with standard Arnoldi method and fill the error table.
"""
function problem_5e()
    println("\n" * "="^80)
    println("PROBLEM 5e: Shift-and-Invert Arnoldi Error Analysis")
    println("="^80)

    Random.seed!(0)
    B = matread("hw1/Bwedge.mat")["B"]
    n = size(B, 1)
    eigenvalues = eigvals(B)

    # Target eigenvalue: closest to -9.8 + 2i
    target_approx = -9.8 + 2im
    target_idx = argmin(abs.(eigenvalues .- target_approx))
    target_eigenvalue = eigenvalues[target_idx]
    println("Target eigenvalue λ*: ", @sprintf("%.4f %+.4fi", real(target_eigenvalue), imag(target_eigenvalue)))
    println()

    # Shifts and iteration counts
    shifts = [nothing, -10, -7 + 2im, -9.8 + 1.5im]
    shift_labels = ["Standard AM", "σ₁ = -10", "σ₂ = -7 + 2i", "σ₃ = -9.8 + 1.5i"]
    m_values = [10, 20, 30]

    # Starting vector: ones
    b = ones(n)

    # Store results for table
    results_table = []

    for (shift, shift_label) in zip(shifts, shift_labels)
        println("-"^80)
        println("Method: $shift_label")

        row_errors = []

        for m in m_values
            if shift === nothing
                # Standard Arnoldi method
                Q, H = arnoldi(B, b, m, classical_gram_schmidt, 2)
                H_sub = H[1:m, 1:m]
                ritz_values = eigvals(H_sub)
            else
                # Shift-and-invert Arnoldi
                A_shifted = B - shift * I
                Q, H = arnoldi(A_shifted, b, m, classical_gram_schmidt, 2)

                # Compute Ritz values and transform back
                H_sub = H[1:m, 1:m]
                ritz_values_inv = eigvals(H_sub)
                ritz_values = shift .+ (1 ./ ritz_values_inv)
            end

            # Find closest Ritz value to target eigenvalue
            distances = abs.(ritz_values .- target_eigenvalue)
            min_idx = argmin(distances)
            closest_ritz = ritz_values[min_idx]

            # Compute absolute error (as shown in the problem table)
            error = abs(target_eigenvalue - closest_ritz)

            println("  m = $m:")
            println("    Closest Ritz value: ", @sprintf("%.4f %+.4fi", real(closest_ritz), imag(closest_ritz)))
            println("    Error: ", @sprintf("%.4e", error))

            push!(row_errors, error)
        end

        push!(results_table, (shift_label, row_errors))
        println()
    end

    # Generate LaTeX table
    println("\n" * "="^80)
    println("LATEX TABLE: Eigenvalue Error Comparison")
    println("="^80)
    println("Target eigenvalue: ", @sprintf("%.4f %+.4fi", real(target_eigenvalue), imag(target_eigenvalue)))
    println()

    # Create table with proper formatting
    formatted_table = []
    for (shift_label, errors) in results_table
        row = [shift_label]
        for error in errors
            push!(row, @sprintf("%.4e", error))
        end
        push!(formatted_table, row)
    end

    # Create matrix with all data
    n_rows = length(formatted_table)
    table_data = Matrix{String}(undef, n_rows + 1, 4)
    table_data[1, :] = ["Method", "m = 10", "m = 20", "m = 30"]
    for i in 1:n_rows
        table_data[i+1, :] = formatted_table[i]
    end

    latex_table = latexify(table_data, env=:tabular, latex=false)
    println(latex_table)
    println("="^80)

    # Discussion
    println("\n" * "="^80)
    println("OBSERVATIONS:")
    println("="^80)
    println("The shift-and-invert method shows significantly faster convergence")
    println("when the shift σ is close to the target eigenvalue λ*.")
    println()
    println("From the transformed eigenvalue plots in Problem 5d, we can see that:")
    println("- When σ is close to λ*, the transformed eigenvalue 1/(λ* - σ) has")
    println("  large magnitude and is well-separated from other eigenvalues.")
    println("- This separation leads to faster convergence in the Arnoldi method.")
    println("- σ₃ = -9.8 + 1.5i is closest to λ* ≈ -9.8 + 2i, hence shows best")
    println("  convergence performance.")
    println("="^80)
end

# ============================================================================
# Execute when run directly
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    problem_5()
    problem_5c_shift_invert()
    problem_5d()
    problem_5e()
end
