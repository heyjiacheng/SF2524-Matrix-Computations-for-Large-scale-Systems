# Problem 5: Compare GMRES and CGN
# CGN = Conjugate Gradients Normal Equations (solves A^T A x = A^T b)

using LinearAlgebra
using MAT
using Plots

include("hw2_functions.jl")

"""
Load data from cgn_illustration.mat
"""
function load_data()
    file = matopen("hw2/cgn_illustration.mat")
    A = read(file, "A")
    b = read(file, "b")
    close(file)
    # Ensure b is a vector
    b = vec(b)
    return A, b
end

"""
Run both methods and collect results
"""
function run_comparison(A, b; maxiter=100, tol=1e-18)
    println("Matrix size: $(size(A))")
    println("Running GMRES...")
    x_gmres, res_gmres, time_gmres = gmres_timed(A, b, tol=tol, maxiter=maxiter)

    println("Running CGN...")
    x_cgn, res_cgn, time_cgn = cgn(A, b, tol=tol, maxiter=maxiter)

    return res_gmres, time_gmres, res_cgn, time_cgn
end

"""
Convert timestamps to elapsed time in seconds
"""
function timestamps_to_seconds(timestamps)
    t0 = timestamps[1]
    return [(t - t0) / 1e9 for t in timestamps]
end

"""
Generate Figure 1: Residual vs Iteration
"""
function plot_iteration_comparison(res_gmres, res_cgn)
    iter_gmres = 0:(length(res_gmres)-1)
    iter_cgn = 0:(length(res_cgn)-1)

    p = plot(
        xlabel="Iteration",
        ylabel="||Ax - b||_2",
        yscale=:log10,
        legend=:topright,
        ylims=(1e-18, 1e3),
        xlims=(0, 80),
        grid=true,
        size=(600, 400)
    )

    plot!(p, iter_gmres, res_gmres,
          label="GMRES",
          linestyle=:dash,
          color=:blue,
          linewidth=1.5)

    plot!(p, iter_cgn, res_cgn,
          label="CGN",
          linestyle=:solid,
          color=:red,
          linewidth=1.5)

    return p
end

"""
Generate Figure 2: Residual vs CPU-time
"""
function plot_time_comparison(res_gmres, time_gmres, res_cgn, time_cgn)
    # Convert to seconds
    t_gmres = timestamps_to_seconds(time_gmres)
    t_cgn = timestamps_to_seconds(time_cgn)

    p = plot(
        xlabel="CPU-time (seconds)",
        ylabel="||Ax - b||_2",
        yscale=:log10,
        legend=:topright,
        ylims=(1e-18, 1e0),
        grid=true,
        size=(600, 400)
    )

    plot!(p, t_gmres, res_gmres,
          label="GMRES",
          linestyle=:dash,
          color=:blue,
          linewidth=1.5)

    plot!(p, t_cgn, res_cgn,
          label="CGN",
          linestyle=:solid,
          color=:red,
          linewidth=1.5)

    return p
end

"""
Main function
"""
function main()
    # Load data
    A, b = load_data()

    # Warm-up run to compile functions
    println("Warm-up run...")
    _ = gmres_timed(A, b, tol=1e-10, maxiter=5)
    _ = cgn(A, b, tol=1e-10, maxiter=5)

    # Actual comparison
    println("\nActual comparison run...")
    res_gmres, time_gmres, res_cgn, time_cgn = run_comparison(A, b, maxiter=100, tol=1e-20)

    # Report results
    println("\nResults:")
    println("GMRES: $(length(res_gmres)-1) iterations, final residual = $(res_gmres[end])")
    println("CGN: $(length(res_cgn)-1) iterations, final residual = $(res_cgn[end])")

    t_gmres = timestamps_to_seconds(time_gmres)
    t_cgn = timestamps_to_seconds(time_cgn)
    println("GMRES total time: $(t_gmres[end]) seconds")
    println("CGN total time: $(t_cgn[end]) seconds")

    # Generate plots
    println("\nGenerating plots...")

    p1 = plot_iteration_comparison(res_gmres, res_cgn)
    savefig(p1, "problem5_iteration.png")
    println("Saved: problem5_iteration.png")

    p2 = plot_time_comparison(res_gmres, time_gmres, res_cgn, time_cgn)
    savefig(p2, "problem5_cputime.png")
    println("Saved: problem5_cputime.png")

    # Display both plots
    display(plot(p1, p2, layout=(1,2), size=(1200, 400)))

    return res_gmres, time_gmres, res_cgn, time_cgn
end

# Run the main function
res_gmres, time_gmres, res_cgn, time_cgn = main()
