
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

# Arnoldi method at each single step:
# Modifies Q and H in place
function arnoldi_step!(A, Q, H, k)
    w = A * Q[:, k]
    w, h = DGS(Q, w, k)
    H[1:k, k] .= h

    # Normalize
    H[k+1, k] = norm(w)
    if H[k+1, k] == 0
        error("Breakdown: Krylov subspace has reached invariant subspace.")
    end

    Q[:, k+1] = w / H[k+1, k]
    return true
end


# ============================================================================
# Double Gram-Schmidt
# ============================================================================
"""
return value:
    w, h - orthogonalized vector(without normalization) and projection coefficients
"""
function DGS(Q, w, k)
    h = zeros(Float64, k)

    for j = 1:k
        h[j] = dot(Q[:, j], w)
        w .-= h[j] * Q[:, j]
    end

    # Second iteration for modification
    for j = 1:k
        δ = dot(Q[:, j], w)
        h[j] += δ
        w .-= δ * Q[:, j]
    end

    return w, h
end


# ============================================================================
# CGN (Conjugate Gradients Normal Equations)
# ============================================================================
"""
    cgn(A, b; tol=1e-10, maxiter=100)

Conjugate Gradient method for Normal Equations (CGN).
Solves Ax = b by applying CG to the normal equations: A^T A x = A^T b.

Key feature: Only uses 2 matrix-vector products per iteration.

# Arguments
- `A`: Matrix (can be non-square, non-symmetric)
- `b`: Right-hand side vector
- `tol`: Tolerance for convergence (default=1e-10)
- `maxiter`: Maximum number of iterations (default=100)

# Returns
- `x`: Solution vector
- `residuals`: Vector of ||Ax - b||_2 at each iteration
- `timestamps`: Vector of CPU timestamps (nanoseconds) at each iteration
"""
function cgn(A, b; tol=1e-10, maxiter=100)
    _, n = size(A)
    T = promote_type(eltype(A), eltype(b))  # Handle complex matrices
    x = zeros(T, n)

    # Initial residual for normal equations: r = A^T b - A^T A x = A^T (b - A x)
    # For x0 = 0, r0 = A^T b
    Ax = A * x                    # First matvec with A
    r_orig = T.(b) - Ax           # Original residual (b - Ax)
    r = A' * r_orig               # Residual for normal equations (A^T r_orig)

    p = copy(r)                   # Initial search direction
    rTr = real(dot(r, r))         # r^H r (use real part for step size)

    residuals = Float64[]
    timestamps = Int64[]

    push!(residuals, norm(r_orig))
    push!(timestamps, time_ns())

    for k = 1:maxiter
        # Compute A^T A p efficiently with 2 matvecs
        Ap = A * p                # First matvec: A * p
        ATAp = A' * Ap            # Second matvec: A^T * (A * p)

        # Step size
        pTATAp = real(dot(p, ATAp))
        if pTATAp == 0
            break
        end
        α = rTr / pTATAp

        # Update solution
        x .+= α .* p

        # Update residuals
        Ax .+= α .* Ap            # Update A*x incrementally (no extra matvec!)
        r_orig = T.(b) - Ax       # Original residual
        r .-= α .* ATAp           # Normal equation residual

        rTr_new = real(dot(r, r))

        # Record residual and timestamp
        push!(residuals, norm(r_orig))
        push!(timestamps, time_ns())

        # Check convergence
        if norm(r_orig) / norm(b) < tol
            break
        end

        # Update search direction
        β = rTr_new / rTr
        p .= r .+ β .* p
        rTr = rTr_new
    end

    return x, residuals, timestamps
end


# ============================================================================
# GMRES with timing
# ============================================================================
"""
    gmres_timed(A, b; tol=1e-10, maxiter=100)

GMRES with timing information for benchmarking.

# Returns
- `x`: Solution vector
- `residuals`: Vector of ||Ax - b||_2 at each iteration
- `timestamps`: Vector of CPU timestamps (nanoseconds) at each iteration
"""
function gmres_timed(A, b; tol=1e-10, maxiter=100)
    n = length(b)
    T = promote_type(eltype(A), eltype(b))  # Handle complex matrices
    r0 = T.(b)
    β = norm(r0)

    residuals = Float64[]
    timestamps = Int64[]

    push!(residuals, β)
    push!(timestamps, time_ns())

    if β < tol
        return zeros(T, n), residuals, timestamps
    end

    Q = zeros(T, n, maxiter + 1)
    H = zeros(T, maxiter + 1, maxiter)
    Q[:, 1] = r0 / β

    e1 = zeros(T, maxiter + 1)
    e1[1] = 1.0

    x = zeros(T, n)

    for k = 1:maxiter
        # Arnoldi step using inline DGS for correct type handling
        w = A * Q[:, k]
        h = zeros(T, k)
        # First GS pass
        for j = 1:k
            h[j] = dot(Q[:, j], w)
            w .-= h[j] * Q[:, j]
        end
        # Second GS pass
        for j = 1:k
            δ = dot(Q[:, j], w)
            h[j] += δ
            w .-= δ * Q[:, j]
        end
        H[1:k, k] .= h

        H[k+1, k] = norm(w)
        if abs(H[k+1, k]) < 1e-14
            # Breakdown - solve and return
            y = H[1:k, 1:k] \ (β * e1[1:k])
            x = Q[:, 1:k] * y
            push!(residuals, norm(b - A * x))
            push!(timestamps, time_ns())
            break
        end

        Q[:, k+1] = w / H[k+1, k]

        # Solve least squares problem
        y = H[1:(k+1), 1:k] \ (β * e1[1:(k+1)])
        x = Q[:, 1:k] * y

        res_norm = norm(b - A * x)
        push!(residuals, res_norm)
        push!(timestamps, time_ns())

        if res_norm / norm(b) < tol
            break
        end
    end

    return x, residuals, timestamps
end

