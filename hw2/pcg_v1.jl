"""
    pcg_v1(A, b, m, T; compute_residual=true)

Performs `m` steps of preconditioned Conjugate Gradient using preconditioner `T`
(not optimized — matches the MATLAB reference implementation).
Returns `(x, rv)` where `rv` is residual history.
"""
function pcg_v1(A, b, m, T; compute_residual=true)
    TT=promote_type(eltype(v),eltype(A));
    x = zeros(TT, length(b))
    r = copy(b)
    p = T \ r
    rv = zeros(TT, m)

    for k in 1:m
        rr = dot(r, T \ r)
        alpha = rr / dot(p, A * p)
        x .+= alpha .* p
        r .-= alpha .* (A * p)
        beta = dot(r, T \ r) / rr
        p = (T \ r) .+ beta .* p

        if compute_residual
            rv[k] = norm(A * x - b) / norm(b)
        end
    end

    return x, rv
end
