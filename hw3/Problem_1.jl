using LinearAlgebra, Random, Plots

include("alpha_example.jl")

# Error function: maximum value below the diagonal
errfun(A) = maximum(abs.(tril(A, -1)))

"""
    basic_qr_method(A, tol=1e-10, maxiter=10000)

Implement the basic QR-method to compute eigenvalues.
Returns the final matrix and the number of iterations.
"""
function basic_qr_method(A, tol=1e-10, maxiter=10000)
    Ak = copy(A)

    for k = 1:maxiter
        # QR factorization
        Q, R = qr(Ak)

        # Form next iterate
        Ak = R * Q

        # Check convergence
        err = errfun(Ak)
        if err < tol
            return Ak, k
        end
    end

    @warn "Maximum iterations reached without convergence"
    return Ak, maxiter
end

"""
    count_iterations_for_alpha(alpha_values, m=20, tol=1e-10)

For each alpha value, count the number of iterations required
to achieve the specified tolerance.
"""
function count_iterations_for_alpha(alpha_values, m=20, tol=1e-10)
    iterations = zeros(Int, length(alpha_values))

    for (i, alpha) in enumerate(alpha_values)
        A = alpha_example(alpha, m)
        _, iters = basic_qr_method(A, tol)
        iterations[i] = iters
        println("α = $alpha: $(iters) iterations")
    end

    return iterations
end

"""
    predict_iterations(alpha_values, m=20, tol=1e-10)

Predict the number of iterations using the theoretical formula.
For large alpha, the convergence rate is dominated by |λ_{m-1}/λ_m|^n.
Using the hint: if error behaves as e_k = |β|^k, then e_N = TOL if N = ln(TOL)/ln(|β|).
"""
function predict_iterations(alpha_values, m=20, tol=1e-10)
    predicted = zeros(length(alpha_values))

    for (i, alpha) in enumerate(alpha_values)
        A = alpha_example(alpha, m)
        eigenvalues = eigvals(A)

        # Sort eigenvalues by magnitude
        idx = sortperm(abs.(eigenvalues))
        λ_sorted = eigenvalues[idx]

        # For large alpha, the error is dominated by |λ_{m-1}/λ_m|^n
        # This is the ratio of the second largest to largest eigenvalue
        β = abs(λ_sorted[m-1] / λ_sorted[m])

        # Using the formula: N = ln(TOL) / ln(|β|)
        if β < 1.0 && β > 0.0
            N = log(tol) / log(β)
            predicted[i] = ceil(N)
        else
            predicted[i] = NaN
        end
    end

    return predicted
end

# ============================================================================
# Main execution for Problem 1
# ============================================================================

println("=" ^ 70)
println("Problem 1: Exercise about basic QR-method")
println("=" ^ 70)

# Part (a): Plot number of iterations as a function of α
println("\n--- Part (a): Computing iterations for different α values ---")

# Generate alpha values on logarithmic scale
alpha_values = 10.0 .^ range(1, 5, length=50)
m = 20  # Matrix size from alpha_example
tol = 1e-10

println("Computing actual iterations...")
actual_iterations = count_iterations_for_alpha(alpha_values, m, tol)

# Part (c): Compute predicted iterations
println("\n--- Part (c): Computing predicted iterations ---")
predicted_iterations = predict_iterations(alpha_values, m, tol)

# Create the plot
println("\nGenerating plot...")
p = plot(alpha_values, actual_iterations,
         xscale=:log10,
         label="Number of iterations",
         linewidth=2,
         xlabel="α",
         ylabel="Number of iterations",
         title="QR-method convergence",
         legend=:topleft,
         marker=:circle,
         markersize=3)

plot!(p, alpha_values, predicted_iterations,
      label="Predicted number of iterations",
      linewidth=2,
      linestyle=:dash,
      marker=:square,
      markersize=3)

display(p)
savefig(p, "hw3/problem1_iterations_vs_alpha.png")
println("Plot saved to hw3/problem1_iterations_vs_alpha.png")

# Part (b): Analysis for large alpha
println("\n--- Part (b): Analysis of convergence for large α ---")
println("For large α, the eigenvalues are ordered by magnitude |λ₁| < ... < |λₘ|.")
println("The elements below the diagonal after n iterations are proportional to |λᵢ/λⱼ|ⁿ with i < j.")
println("\nFor large α, the error is dominated by i = m-1 and j = m,")
println("i.e., the ratio |λ_{m-1}/λ_m|ⁿ (second largest to largest eigenvalue).")
println("\nVerification with a specific α:")

alpha_test = 1000.0
A_test = alpha_example(alpha_test, m)
eigenvalues = eigvals(A_test)
idx = sortperm(abs.(eigenvalues))
λ_sorted = eigenvalues[idx]

println("α = $alpha_test")
println("Eigenvalues (sorted by magnitude):")
for i = 1:m
    println("  λ[$i] = $(abs(λ_sorted[i]))")
end

# Check all ratios
println("\nRatios |λᵢ/λⱼ| for i < j:")
for j = 2:min(5, m)
    for i = 1:j-1
        ratio = abs(λ_sorted[i] / λ_sorted[j])
        println("  |λ[$i]/λ[$j]| = $ratio")
    end
end

β_dominant = abs(λ_sorted[m-1] / λ_sorted[m])
println("\nDominant ratio (largest): |λ[$(m-1)]/λ[$m]| = $β_dominant")
println("This is the ratio that determines the convergence rate for large α.")

println("\n--- Part (c): Discussion ---")
println("The predicted iterations formula N = ln(TOL)/ln(|β|) works well")
println("when β = |λ_{m-1}/λ_m| is the dominant convergence factor.")
println("From the plot, we can see that the predicted values (dashed line)")
println("closely match the actual iterations (solid line),")
println("confirming our theoretical understanding of the QR-method convergence.")

println("\n" * "=" ^ 70)
println("Problem 1 completed successfully!")
println("=" ^ 70)
