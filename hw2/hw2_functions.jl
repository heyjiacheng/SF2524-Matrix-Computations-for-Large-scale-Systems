
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

