using LinearAlgebra
using SparseArrays
using Random
using Plots
using PyPlot

include("hw2_functions.jl")

# Implement GMRES based on the Arnoldi method using double Gram-Schmidt.
function GMRES(A, b; tol=1e-8, nmax=100, x0=zeros(length(b)), x_exact=nothing)
    r0 = b - A * x0
    β = norm(r0)
    xk = x0
    rel_residuals = []
    rel_errors = []

    if β < tol
        return x0, 0
    end

    v1 = r0 / β
    Q = zeros(ComplexF64, length(b), nmax + 1)
    H = zeros(ComplexF64, nmax + 1, nmax)
    Q[:, 1] = v1

    e1 = zeros(ComplexF64, nmax + 1)
    e1[1] = 1.0

    for k = 1:nmax
        continue_flag = arnoldi_step!(A, Q, H, k)
        if !continue_flag
            break
        end
        # Solve the least squares problem
        y = H[1:(k+1), 1:k] \ (β * e1[1:(k+1)])
        xk = x0 + Q[:, 1:k] * y

        push!(rel_errors, norm(x_exact - xk) / norm(x_exact))
        push!(rel_residuals, norm(b - A * xk) / norm(b))

        # Check convergence
        if rel_residuals[end] < tol
            return xk, rel_errors, rel_residuals
        end
    end
    
    return xk, rel_errors, rel_residuals
end

# Generate matrix A and vector
function generate_data(n; alpha=5.0, seed=5)
    Random.seed!(seed)
    A = sprand(n, n, 0.5)
    A += alpha * sparse(I, n, n)
    A = A ./ opnorm(A,1)
    b = rand(n, 1)
    return A, b
end

# Visualization using semilogy
function semilogy_plot(error_list, residual_list, x_exact; title_str="GMRES Convergence")
    iterations = 1:length(error_list)
    
    figure()
    semilogy(iterations, error_list, color="black", linestyle="--", label="Relative Error")
    semilogy(iterations, residual_list, color= "red", linestyle="-", label="Relative Residual")
    semilogy(iterations, x_exact, color= "blue", linestyle= "--", label="theory prediction")

    xlabel("Iteration")
    ylabel("Error")

    grid=true
    title(title_str)
    legend()
    PyPlot.savefig("gmres_convergence.png")

end


function main()
    A, b = generate_data(100, alpha=5.0, seed=5)
    x_exact = A \ b
    x_computed, error_list, residual_list = GMRES(A, b, tol=1e-20, nmax=100, x0=zeros(length(b)), x_exact=x_exact)
    semilogy_plot(error_list, residual_list, x_exact, title_str="GMRES Convergence for n=100")
end

main()