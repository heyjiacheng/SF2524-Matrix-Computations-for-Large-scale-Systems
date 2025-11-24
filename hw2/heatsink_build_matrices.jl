using MAT
using LinearAlgebra

# --- Load data ---
data = matread("hw2/heatsink_exercise_ref1.mat")

K         = data["K"]          # full stiffness
F0        = data["F"]          # base load vector (without Gaussian)
coords    = data["coords"]     # N x 3
tri       = convert.(Int64,data["tri"])        # M x 3
baseNodes = convert.(Int64,data["baseNodes"])  # indices of base nodes (y ≈ ymin)
topNodes  = convert.(Int64,data["topNodes"])   # indices of top nodes  (y ≈ ymax)

N = size(coords, 1)

# --- Parameters for Gaussian base temperature ---

Tinf     = 293.15   # K, ambient
T0       = 363.15   # K, peak at center (e.g. 90°C)
sigma    = 0.03     # m, Gaussian width
x_center = 0.0      # m
z_center = 0.0      # m

# Extract base node positions
x_base = coords[baseNodes, 1]
z_base = coords[baseNodes, 3]

# Gaussian temperature profile at base
dx2 = (x_base .- x_center).^2
dz2 = (z_base .- z_center).^2
Tbase = Tinf .+ (T0 - Tinf) .* exp.(-(dx2 .+ dz2) ./ (2 * sigma^2))

# --- Build Dirichlet-constrained system and solve ---

# Initialize full temperature vector with ambient
T = fill(Tinf, N)
T[baseNodes] .= Tbase

fixed = falses(N)
fixed[baseNodes] .= true
free = .!fixed

# Reduced system: A * Tfree = Ff - Kfc * Tfixed
A   = K[free, free]
Kfc = K[free, fixed]
Ff  = F0[free]

rhs = Ff - Kfc * T[fixed]

# Solve
x = A \ rhs
