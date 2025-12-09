using Statistics  # for median

function create_adjacency(X, cutoff)

    # X is D x N  (each column = one image)
    N = size(X, 2)

    # --- Pairwise Euclidean distances between columns ---
    X2 = sum(X.^2, dims = 1)                        # 1 x N, squared norms of columns
    dist2 = X2 .+ X2' .- 2 .* (X' * X)             # N x N, squared distances
    dist2 = max.(dist2, 0)                         # numerical safety
    Dmat  = sqrt.(dist2)                           # Euclidean distances

    # --- Choose sigma (example: median of off-diagonal distances) ---
    iu    = triu(trues(N, N), 1)                   # upper triangle (i < j)
    allD  = Dmat[iu]
    sigma = median(allD)                           # or whatever you prefer

    # --- Gaussian weights ---
    W = exp.(-(Dmat.^2) ./ (2 * sigma^2))          # fully connected

    W[1:(N+1):end] .= 0                            # optional: remove self-loops

    W[W .< cutoff] .= 0

    A = W
    return A
end
