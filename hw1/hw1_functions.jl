# ============================================================================
# Core Eigenvalue Iteration Methods
# ============================================================================

"""
    rayleigh_quotient(A, v)

Compute the Rayleigh quotient: R(A, v) = (v' * A * v) / (v' * v)
"""
function rayleigh_quotient(A, v)
    return dot(v, A * v) / dot(v, v)
end

"""
    power_method(A, v0; tol=1e-10)

Find the largest eigenvalue using the power method.
Returns the error history.
"""
function power_method(A, v0; tol=1e-10)
    v = v0 / norm(v0)
    eigenvalue_exact = maximum(abs, eigvals(A))
    eigenvalue_error = Float64[]

    eigenvalue_old = rayleigh_quotient(A, v)
    v_new = A * v
    v_new /= norm(v_new)
    eigenvalue_new = rayleigh_quotient(A, v_new)

    iter = 0
    while abs(eigenvalue_new - eigenvalue_old) > tol
        eigenvalue_old = eigenvalue_new
        push!(eigenvalue_error, abs(eigenvalue_new - eigenvalue_exact) / abs(eigenvalue_exact))

        v = v_new
        v_new = A * v
        v_new /= norm(v_new)
        eigenvalue_new = rayleigh_quotient(A, v_new)
        iter += 1
    end

    println("\nPower Method:")
    println("  Largest eigenvalue: ", rayleigh_quotient(A, v_new))
    println("  Converged in $iter iterations")

    return eigenvalue_error
end

"""
    inverse_iteration(A, v0, shift; tol=1e-10, max_iter=100)

Find the eigenvalue closest to `shift` using inverse iteration.
Returns the error history.
"""
function inverse_iteration(A, v0, shift; tol=1e-10, max_iter=100)
    v = v0 / norm(v0)
    eigenvalue_error = Float64[]
    A_shifted = A - shift * I

    # Find target eigenvalue (closest to shift)
    all_eigenvalues = eigvals(A)
    target_idx = argmin(abs.(all_eigenvalues .- shift))
    eigenvalue_exact = all_eigenvalues[target_idx]

    eigenvalue_old = rayleigh_quotient(A, v)

    for iter in 1:max_iter
        # Solve (A - shift*I) * v_new = v
        v_new = A_shifted \ v
        v_new /= norm(v_new)

        eigenvalue_new = rayleigh_quotient(A, v_new)
        push!(eigenvalue_error, abs(eigenvalue_new - eigenvalue_exact) / abs(eigenvalue_exact))

        if abs(eigenvalue_new - eigenvalue_old) < tol
            println("\nInverse Iteration (shift μ = $shift):")
            println("  Estimated eigenvalue: $eigenvalue_new")
            println("  Exact eigenvalue: $eigenvalue_exact")
            println("  Converged in $iter iterations")
            return eigenvalue_error
        end

        eigenvalue_old = eigenvalue_new
        v = v_new
    end

    println("\nInverse Iteration: Maximum iterations reached")
    return eigenvalue_error
end

"""
    rayleigh_quotient_iteration(A, v0; tol=1e-10, name="Matrix")

Find an eigenvalue using Rayleigh Quotient Iteration (cubic convergence).
Returns the error history.
"""
function rayleigh_quotient_iteration(A, v0; tol=1e-10, name="Matrix")
    v = v0 / norm(v0)
    eigenvalue_old = rayleigh_quotient(A, v)

    v_new = (A - eigenvalue_old * I) \ v
    v_new /= norm(v_new)
    eigenvalue_new = rayleigh_quotient(A, v_new)

    # Find closest exact eigenvalue
    all_eigenvalues = eigvals(A)
    eigenvalue_exact = all_eigenvalues[argmin(abs.(all_eigenvalues .- eigenvalue_new))]

    eigenvalue_error = Float64[]
    iter = 0

    while abs(eigenvalue_new - eigenvalue_old) > tol
        push!(eigenvalue_error, abs(eigenvalue_new - eigenvalue_exact) / abs(eigenvalue_exact))
        eigenvalue_old = eigenvalue_new

        v_new = (A - eigenvalue_old * I) \ v
        v_new /= norm(v_new)
        eigenvalue_new = rayleigh_quotient(A, v_new)

        v = v_new
        iter += 1
    end

    println("\nRayleigh Quotient Iteration ($name):")
    println("  Estimated eigenvalue: $eigenvalue_new")
    println("  Exact eigenvalue: $eigenvalue_exact")
    println("  Converged in $iter iterations")

    return eigenvalue_error
end

# ============================================================================
# Gram-Schmidt Orthogonalization Variants
# ============================================================================

"""
    classical_gram_schmidt(Q, w, k, cycles=1)

Classical Gram-Schmidt orthogonalization (for-loop version).
"""
function classical_gram_schmidt(Q, w, k, cycles=1)
    h = zeros(ComplexF64, k)
    z = w

    for cycle in 1:cycles
        for j in 1:k
            h_j = dot(Q[:, j], z) / dot(Q[:, j], Q[:, j])
            z -= h_j * Q[:, j]
            h[j] += h_j
        end
    end

    β = norm(z)
    return h, β, z
end

"""
    modified_gram_schmidt(Q, w, k, cycles=1)

Modified Gram-Schmidt orthogonalization (sequential version).
"""
function modified_gram_schmidt(Q, w, k, cycles=1)
    h = zeros(ComplexF64, k)
    z = w

    for j in 1:k
        h[j] = dot(Q[:, j], z) / dot(Q[:, j], Q[:, j])
        z -= h[j] * Q[:, j]
    end

    β = norm(z)
    return h, β, z
end

"""
    CGS_MV(Q, w, k, cycles=1)

Classical Gram-Schmidt with Matrix-Vector product (MV version).
Computes orthogonalization using matrix-vector operations for efficiency.
"""
function CGS_MV(Q, w, k, cycles=1)
    h = zeros(ComplexF64, k)
    z = w

    for cycle in 1:cycles
        # Matrix-vector product: h_new = Q[:, 1:k]' * z
        Qk = @view Q[:, 1:k]
        h_new = Qk' * z
        # Update z: z = z - Q[:, 1:k] * h_new
        z = z - Qk * h_new
        h = h + h_new
    end

    β = norm(z)
    return h, β, z
end

"""
    CGS_for(Q, w, k, cycles=1)

Classical Gram-Schmidt with for-loop version.
Uses explicit for-loop over vectors.
"""
function CGS_for(Q, w, k, cycles=1)
    h = zeros(ComplexF64, k)
    z = w

    for cycle in 1:cycles
        for j in 1:k
            h_j = dot(Q[:, j], z)
            z = z - h_j * Q[:, j]
            h[j] += h_j
        end
    end

    β = norm(z)
    return h, β, z
end

"""
    MGS_for(Q, w, k, cycles=1)

Modified Gram-Schmidt with for-loop version.
Performs orthogonalization sequentially, updating z after each projection.
"""
function MGS_for(Q, w, k, cycles=1)
    h = zeros(ComplexF64, k)
    z = w

    for j in 1:k
        h[j] = dot(Q[:, j], z)
        z = z - h[j] * Q[:, j]
    end

    β = norm(z)
    return h, β, z
end

"""
    cgs_mv(Q, w, k, cycles=1)

Classical Gram-Schmidt with matrix-vector operations (optimized version).
Alias for CGS_MV for compatibility.
"""
function cgs_mv(Q, w, k, cycles=1)
    return CGS_MV(Q, w, k, cycles)
end

# ============================================================================
# Arnoldi Method
# ============================================================================

"""
    arnoldi(A, b, m, gs_func, cycles=1)

Arnoldi method for Krylov subspace iteration.

# Arguments
- `A`: Matrix
- `b`: Starting vector
- `m`: Number of iterations
- `gs_func`: Gram-Schmidt function (classical_gram_schmidt or modified_gram_schmidt)
- `cycles`: Number of re-orthogonalization cycles (default=1)

# Returns
- `Q`: Orthonormal basis matrix (n × (m+1))
- `H`: Upper Hessenberg matrix ((m+1) × m)
"""
function arnoldi(A, b, m, gs_func, cycles=1)
    n = length(b)
    Q = zeros(ComplexF64, n, m + 1)
    H = zeros(ComplexF64, m + 1, m)
    Q[:, 1] = b / norm(b)

    for k = 1:m
        w = A * Q[:, k]
        h, β, z = gs_func(Q, w, k, cycles)
        H[1:(k+1), k] = [h; β]
        Q[:, k+1] = z / β
    end

    return Q, H
end

"""
    arnoldi_inverse(A, b, m, gs_func, cycles=1)

Arnoldi method applied to A^(-1) (inverse Arnoldi).
"""
function arnoldi_inverse(A, b, m, gs_func, cycles=1)
    n = length(b)
    Q = zeros(ComplexF64, n, m + 1)
    H = zeros(ComplexF64, m + 1, m)
    Q[:, 1] = b / norm(b)

    for k = 1:m
        w = A \ Q[:, k]
        h, β, z = gs_func(Q, w, k, cycles)
        H[1:(k+1), k] = [h; β]
        Q[:, k+1] = z / β
    end

    return Q, H
end

# ============================================================================
# Krylov Basis Construction
# ============================================================================

"""
    construct_krylov_basis(A, b, m)

Construct Km matrix with normalized power method iterates.
Km = [b/||b||, Ab/||Ab||, A²b/||A²b||, ..., A^(m-1)b/||A^(m-1)b||]

# Note
This basis is NOT orthogonalized and becomes numerically rank-deficient
for large m due to repeated multiplication by A.
"""
function construct_krylov_basis(A, b, m)
    n = length(b)
    Km = zeros(n, m)

    v = b / norm(b)
    Km[:, 1] = v

    for k = 2:m
        v = A * v
        v /= norm(v)
        Km[:, k] = v
    end

    return Km
end

"""
    eigenvalue_approximation_krylov(A, Km)

Compute eigenvalue approximation using equation (2):
    μw = (Km' * Km)^(-1) * Km' * A * Km * w

This solves the generalized eigenvalue problem:
    (Km' * A * Km) * w = μ * (Km' * Km) * w

# Returns
Vector of Ritz values (eigenvalue approximations).

# Note
This method works without explicit orthogonalization but suffers from
numerical instability when Km' * Km becomes ill-conditioned.
"""
function eigenvalue_approximation_krylov(A, Km)
    m = size(Km, 2)

    # Solve: (Km' * Km)^(-1) * (Km' * A * Km) * w = μ * w
    KmT_Km = Km' * Km
    KmT_A_Km = Km' * (A * Km)
    M = KmT_Km \ KmT_A_Km

    return eigvals(M)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    compare_eigenvalues_ritzvalues(eigenvalues, ritzvalues)

Compare exact eigenvalues with Ritz values and compute relative errors.
Returns array of tuples: (closest_eigenvalue, relative_error).
"""
function compare_eigenvalues_ritzvalues(eigenvalues, ritzvalues)
    errors = []
    for ritz in ritzvalues
        min_index = argmin(abs.(eigenvalues .- ritz))
        closest_eigenvalue = eigenvalues[min_index]
        relative_error = abs(closest_eigenvalue - ritz) / abs(closest_eigenvalue)
        push!(errors, (closest_eigenvalue, relative_error))
    end
    return errors
end

"""
    benchmark_gs(A, b, m, gs_func, cycles=1; n_trials=3)

Benchmark a specific Gram-Schmidt configuration.

# Returns
- `time`: Minimum execution time over n_trials
- `orth`: Orthogonality measure ||Q'Q - I||
"""
function benchmark_gs(A, b, m, gs_func, cycles=1; n_trials=3)
    # Warm-up run
    Q, H = arnoldi(A, b, m, gs_func, cycles)

    # Multiple timed runs for better accuracy
    times = zeros(n_trials)
    for i in 1:n_trials
        times[i] = @elapsed Q, H = arnoldi(A, b, m, gs_func, cycles)
    end

    # Final run to measure orthogonality
    Q, H = arnoldi(A, b, m, gs_func, cycles)
    Qm = Q[:, 1:m]
    orth = norm(Qm' * Qm - I)

    return minimum(times), orth
end

