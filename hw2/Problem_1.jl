using LinearAlgebra
using SparseArrays
using Statistics
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
function semilogy_plot(error_list, residual_list, theory_list; title_str="GMRES Convergence_Localization Disk")
    iterations = 1:length(error_list)
    
    figure()
    semilogy(iterations, error_list, color="black", linestyle="--", label="Relative Error")
    semilogy(iterations, residual_list, color= "red", linestyle="-", label="Relative Residual")
    semilogy(iterations, theory_list, color= "blue", linestyle= "--", label="theory prediction by Localization Disk")

    xlabel("Iteration")
    ylabel("Error")

    PyPlot.grid(true)
    title(title_str)
    legend()
    PyPlot.savefig(title_str * ".png")

end

# To calculate the Gershgorin disks and plot eigenvalues and disks
function ger_disk(A)
    c = tr(A) / size(A, 1)
    # We choose to use the outlier version of Gershgorin disks
    R1 = sort([sum(abs.(A[i, :])) - abs(A[i, i]) for i in 1:size(A, 1)])
    R = quantile(R1, 0.9)
    return c, R
end

function loc_disk(A)
    λ = eigvals(Matrix(A))
    centroid = sum(λ) / length(λ)
    distances = sort(abs.(λ .- centroid))
    radius = distances[end-1] #Second largest to consider the outlier
    return centroid, radius
end

function plot_ger_disk(A, a)
    λ = eigvals(Matrix(A))
    c, R = loc_disk(A)

    θ = LinRange(0, 2π, 100)
    x_circle = real(c) .+ R * cos.(θ)
    y_circle = imag(c) .+ R * sin.(θ)

    Plots.scatter(real.(λ), imag.(λ), color=:black, marker = :cross, label="Eigenvalues")
    Plots.plot!(x_circle, y_circle, color=:red, label="Localization Disk", xlabel="Re λ", ylabel="Im λ", title="α = $(a)")
    xlims!(-0.2, 1.2)
    ylims!(-0.2, 0.2)

    Plots.savefig("eigenvalues_α=$(a).png")
end

function gmres_measure(A, b, m)
    t = @elapsed x = GMRES(A, b, nmax=m)
    resnorm = norm(A*x - b)
    return t, resnorm
end

function main()
    alpha = [1.0, 5.0, 10.0, 100.0]
    nmax = 100
    for a in alpha
        A, b = generate_data(100, alpha=a, seed=5)
        x_exact = A \ b
        # question(a): GMRES convergence plot
        x_computed, error_list, residual_list = GMRES(A, b, tol=1e-20, nmax=100, x0=zeros(length(b)), x_exact=x_exact)
        c, R = loc_disk(A)
        ρ = R / abs(c)
        theory_list = ρ .^ (1:nmax)
        semilogy_plot(error_list, residual_list, theory_list, title_str="Localization Disk_GMRES Convergence for α=$(a)")
        
        # question(b): Plot the eigenvalues of A for different α
        plot_ger_disk(A, a)
    end

    # # question(c): Time and residual norm measurement
    # Attention: If we want to run this part, we need to change the return value of GMRES to return only xk.
    
    # alphas = [1.0, 100.0]
    # n_list = [200, 500, 1000]
    # m_list = [5, 10, 20, 50, 100]
    # for a in alphas
    #     println("Alpha = $(a)")
    #     for n in n_list
    #         A, b = generate_data(n, alpha=a, seed=5)
    #         for m in m_list
    #             # warm-up
    #             x = GMRES(A, b, nmax=m)
    #             t, resnorm = gmres_measure(A, b, m)
    #             println("n=$(n), m=$(m): time=$(t), residual norm=$(resnorm)")
    #         end

    #         # backslash
    #         t_bs = @elapsed x_bs = A \ b
    #         resnorm_bs = norm(A*x_bs - b)
    #         println("n=$(n), backslash: time=$(t_bs), residual norm=$(resnorm_bs)")
    #     end
    # end

    
end

main()