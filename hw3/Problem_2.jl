using LinearAlgebra, Printf, Latexify, DataFrames

# Include the necessary files
include("naive_hessenberg_red.jl")
include("alpha_example.jl")

println("="^60)
println("Problem 2: Hessenberg reduction and shifted QR-method")
println("="^60)

# ============================================================================
# Part (b): Naive Hessenberg reduction
# ============================================================================
println("\n" * "="^60)
println("Part (b): Naive Hessenberg reduction (already completed)")
println("="^60)
println("The naive_hessenberg_red.jl file has been completed.")
println("Testing with a 5x5 random matrix:")

A_test = rand(5,5)
H_test = naive_hessenberg_red(copy(A_test))
println("\nOriginal matrix A:")
display(A_test)
println("\n\nHessenberg form H:")
display(H_test)
println("\n\nVerification - maximum element below first off-diagonal:")
below_offdiag = maximum(abs.(tril(H_test, -2)))
println("  max|H[i,j]| for i > j+1: ", below_offdiag)
println("  (should be very small, ~1e-15)")

# Verify eigenvalues are preserved
ev_A = sort(eigvals(A_test), by=abs)
ev_H = sort(eigvals(H_test), by=abs)
println("\nEigenvalue preservation error: ", norm(ev_A - ev_H))


# ============================================================================
# Part (c): Efficient Algorithm 2 for Hessenberg reduction
# ============================================================================
println("\n\n" * "="^60)
println("Part (c): Efficient Algorithm 2 implementation")
println("="^60)

function efficient_hessenberg(A)
    # Implements Algorithm 2 from lecture notes
    # More efficient by avoiding explicit matrix formation
    n = size(A, 1)
    A = copy(A)  # Don't modify input

    for k = 1:n-2
        # Extract column k from row k+1 onwards
        x = A[k+1:n, k]

        # Compute Householder reflector using (3.4)
        # u_k such that P_k x = ||x|| e_1
        e1 = zeros(n-k)
        e1[1] = 1.0
        z = x + sign(x[1]) * norm(x) * e1
        u = z / norm(z)

        # Compute P_k A: A[k+1:n, k:n] := A[k+1:n, k:n] - 2u(u^* A[k+1:n, k:n])
        # This is more efficient than forming P explicitly
        A[k+1:n, k:n] = A[k+1:n, k:n] - 2 * u * (u' * A[k+1:n, k:n])

        # Compute P_k A P_k^*: A[1:n, k+1:n] := A[1:n, k+1:n] - 2(A[1:n, k+1:n]u)u^*
        A[1:n, k+1:n] = A[1:n, k+1:n] - 2 * (A[1:n, k+1:n] * u) * u'
    end

    return A
end

println("\nTesting efficient algorithm with same 5x5 matrix:")
H_efficient = efficient_hessenberg(A_test)
println("Difference between naive and efficient methods:")
println("  ||H_naive - H_efficient|| = ", norm(H_test - H_efficient))
println("  (should be very small)")


# ============================================================================
# Part (c): Timing comparison
# ============================================================================
println("\n\n" * "="^60)
println("Part (c): Timing comparison")
println("="^60)

println("\nComparing CPU time for different matrix sizes m:")
println("Using A = alpha_example(1, m)")
println("\n" * "-"^60)
@printf("%-10s %-25s %-25s\n", "m", "CPU-time Algorithm 2", "CPU-time naive")
println("-"^60)

sizes = [10, 100, 200, 300, 400]
times_efficient = zeros(length(sizes))
times_naive = zeros(length(sizes))

for (idx, m) in enumerate(sizes)
    local A_local = alpha_example(1, m)

    # Time efficient algorithm (multiple runs for better accuracy)
    n_runs = (m <= 100) ? 10 : 3
    time_eff = @elapsed for i = 1:n_runs
        H_eff = efficient_hessenberg(A_local)
    end
    times_efficient[idx] = time_eff / n_runs

    # Time naive algorithm
    time_naive = @elapsed for i = 1:n_runs
        H_naive = naive_hessenberg_red(A_local)
    end
    times_naive[idx] = time_naive / n_runs

    @printf("%-10d %-25.6f %-25.6f\n", m, times_efficient[idx], times_naive[idx])
end

println("-"^60)

# Create LaTeX table for part (c)
df_timing = DataFrame(
    m = sizes,
    Algorithm_2 = times_efficient,
    Naive = times_naive
)

println("\n\nLaTeX table for part (c):")
println("="^60)
println(latexify(df_timing, env=:table, fmt="%.6f",
                 latex=false, booktabs=true))

println("\nObservation: The efficient Algorithm 2 is significantly faster,")
println("especially for larger matrices, as it avoids forming the full")
println("Householder matrices explicitly.")


# ============================================================================
# Part (d): Shifted QR method
# ============================================================================
println("\n\n" * "="^60)
println("Part (d): Shifted QR method")
println("="^60)

function shifted_qr_step(A, sigma)
    # Performs one step of shifted QR method
    # Input: matrix A, shift sigma
    # Output: H_bar (result after one QR step)

    n = size(A, 1)

    # Step 1: Compute QR factorization of (A - sigma*I)
    Q, R = qr(A - sigma * I)

    # Step 2: Compute H_bar = R*Q + sigma*I
    H_bar = R * Q + sigma * I

    return H_bar
end

println("\nFor the 2x2 matrix A = [3 2; ε 1]")
println("Computing |h̄₂,₁| after one shifted QR step")
println("\n" * "-"^70)
@printf("%-15s %-25s %-25s\n", "ε", "|h̄₂,₁| (σ=0)", "|h̄₂,₁| (σ=a₂,₂=1)")
println("-"^70)

epsilon_values = [0.4, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0.0]
h21_sigma0_vals = zeros(length(epsilon_values))
h21_sigma1_vals = zeros(length(epsilon_values))

for (idx, eps) in enumerate(epsilon_values)
    local A_local = [3.0 2.0; eps 1.0]

    # Shift σ = 0
    H_bar_0 = shifted_qr_step(A_local, 0.0)
    h21_sigma0 = abs(H_bar_0[2, 1])

    # Shift σ = a₂,₂ = 1
    H_bar_1 = shifted_qr_step(A_local, 1.0)
    h21_sigma1 = abs(H_bar_1[2, 1])

    h21_sigma0_vals[idx] = h21_sigma0
    h21_sigma1_vals[idx] = h21_sigma1

    @printf("%-15g %-25.4e %-25.4e\n", eps, h21_sigma0, h21_sigma1)
end

println("-"^70)

# Create LaTeX table for part (d)
df_shifted = DataFrame(
    epsilon = epsilon_values,
    sigma_0 = h21_sigma0_vals,
    sigma_1 = h21_sigma1_vals
)

println("\n\nLaTeX table for part (d):")
println("="^60)
println(latexify(df_shifted, env=:table, fmt="%.4e",
                 latex=false, booktabs=true))

println("\n" * "="^60)
println("Interpretation of results:")
println("="^60)
println("""
1. When σ = 0 (unshifted QR method):
   - The off-diagonal element |h̄₂,₁| shows the convergence behavior
   - For this matrix, it converges but not optimally

2. When σ = a₂,₂ = 1 (shifted by the (2,2) element):
   - The convergence is much faster (smaller |h̄₂,₁|)
   - This is because we're shifting by a value close to an eigenvalue
   - The eigenvalues of A are approximately 1 and 3 when ε is small
   - Shifting by σ = 1 (which is exactly λ₂ when ε = 0) makes the
     method converge rapidly to isolate this eigenvalue

3. As ε → 0:
   - With σ = 1, we see |h̄₂,₁| → 0 very rapidly (quadratic convergence)
   - This demonstrates that shifting with a good approximation to an
     eigenvalue greatly accelerates convergence

4. When σ = 0, the shifted QR method reduces to the standard QR method,
   which is equivalent to applying power iteration with the deflation
   strategy.

5. The better choice of σ in this case is σ = a₂,₂ = 1, as it provides
   much faster convergence to the eigenvalue structure.
""")

println("\n" * "="^60)
println("Problem 2 completed!")
println("="^60)
