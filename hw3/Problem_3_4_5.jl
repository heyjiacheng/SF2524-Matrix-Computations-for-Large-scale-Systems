using IterativeSolvers
using Optim
using MAT
using Plots
using SparseArrays
using Random
using MatrixDepot
using LinearAlgebra
using Latexify
using StatsBase

function naive_hessenberg_red(A)

    n = size(A, 1)
    for k = 1:(n-2)
        x = A[k+1:end, k]

        e1 = zeros(length(x))
        e1[1] = norm(x)
        
        u = x - e1
        if norm(u) != 0
            u = u / norm(u)
        end
        
        P1 = I - 2 * (u * u')
        P = Matrix{Float64}(I, n, n)
        P[k+1:end, k+1:end] .= P1

        A = P * A * P'  
    end
    return A
end

function Algo_2(A)
    n = size(A, 1)
    for k in 1:(n-2)
        x = A[k+1:n, k]
        e1 = zeros(size(x))
        e1[1] = 1
        u = x - norm(x) * e1
        if norm(u) != 0
            u = u / norm(u)
        end
        A[k+1:n, k:n] .= A[k+1:n, k:n] - 2 * u * (u' * A[k+1:n, k:n])
        A[1:n, k+1:n] .= A[1:n, k+1:n] - 2 * (A[1:n, k+1:n] * u) * u'
    end
    return A
end

function alpha_example(alpha,m)
    # Usage:
    #  A=alpha_example(alpha,m)
    #
    # Generates a matrix of size m (which is 20 unless
    # it is specified). alpha is a parameter which should be
    # in the interval (0,infty).
    #
    # The basic QR-method will be have in very different ways
    # for large and small alpha
    #
  if (~(alpha>0))
    error("alpha must be positive")
  end


  Random.seed!(0)
  a::Float64=2;
  d=(a.^(0:(m-1)))
  d[end-1]=d[end]*(0.99-1/(5*alpha))
  B=randn(m,m)
  A=B\(diagm(0=>d[:])*B)
  return A
end

A=alpha_example(0.5,10)


function basic_QR(A, max_iter=1000, tol=1e-10, shift = 0)
    rrfun = A -> maximum(maximum(abs.(tril(A, -1))))
    Q, R = qr(A-shift*I)
    A = R * Q
    U = Q
    j = 1
    while true
        Q, R = qr(A)
        A = R * Q + shift * I
        U = U * Q
        j+=1
        if rrfun(A) < tol
            break
        end
    end
    return A, U, j
    
end

function moving_average(data, window_size)
    return [mean(data[max(1, i-window_size+1):i]) for i in 1:length(data)]
end

function plot_with_moving_average(N_max, times_schur, times_naive, window_size)
    plot(
        1:N_max, 
        times_schur, 
        label="Schur-Parlett", 
        xlabel="N", 
        ylabel="Time",
        alpha=0.5
    )
    schur_moving_avg = moving_average(times_schur, window_size)
    plot!(1:N_max, schur_moving_avg, label="Schur-Parlett Moving Average", linestyle=:dash)
    
    plot!(
        1:N_max, 
        times_naive, 
        label="Naive", 
        xlabel="N", 
        ylabel="Time",
        alpha=0.5
    )
    naive_moving_avg = moving_average(times_naive, window_size)
    plot!(1:N_max, naive_moving_avg, label="Naive Moving Average", linestyle=:dash)
    
    savefig("1/Ex3_b.png")
end

function schur_parlett(A, f)
    # The Schur-Parlett method. The function handle f is a scalar-valued function.
    T, Q = schur(Matrix{ComplexF64}(A));  # Complex Schur form (works for complex eigenvalues)
    n = size(A, 1);
    F = zeros(ComplexF64, n, n);

    # Apply the function f to the diagonal entries of T
    for i = 1:n
        F[i, i] = f(T[i, i]);
    end

    # Compute off-diagonal elements using Parlett recurrence
    for p = 1:n-1
        for i = 1:(n-p)
            j = i + p;
            s = T[i, j] * (F[j, j] - F[i, i]);
            for k = i+1:j-1
                s += T[i, k] * F[k, j] - F[i, k] * T[k, j];
            end
            if abs(T[j, j] - T[i, i]) > 1e-12
                F[i, j] = s / (T[j, j] - T[i, i]);
            else
                error("Numerical instability: Denominator close to zero.");
            end
        end
    end
    F = Q * F * Q';
    return F
end

function naiveexp(A, n)
    A0 = A
    for i = 1:n
        A = A * A0
    end
    return A
end

function Ex1()
    alphas = 10 .^ range(log10(10), log10(1e5), length=50)
    m = 20
    iter = []
    theo_iter = []
    for alpha in alphas
        A = alpha_example(alpha, m)
        eigenv = eigvals(A)
        eigenv = sort(eigenv)
        eigenv_quotient = []
        for i in 1:(length(eigenv)-1)
            push!(eigenv_quotient, eigenv[i]/eigenv[i+1])
        end
        push!(theo_iter, -10/log10(maximum(eigenv_quotient)))
        A, U, i = basic_QR(A)
        push!(iter, i)
    end

    plot(
        alphas, 
        iter, 
        label="Number of iterations", 
        xlabel="alpha", 
        ylabel="Number of iterations", 
        xaxis=:log, 
        marker=:circle, markersize=2)
    savefig("1/Ex1.png")
    plot!(
        alphas, theo_iter,
        label = "Theoretical number of iterations",
        marker = :circle,
        markersize = 2,
    )
    y2 = twinx()
    plot!(y2,
        alphas, iter ./ theo_iter,
        label = "Ratio",
        ylabel = "Measured / Theoretical",
        xaxis = :log,
        marker = :circle,
        markersize = 2,
        ylims = (0, 2),
        color = :red,
    )

    # Combine the two plots into one figure
    
    savefig("1/Ex3.png")
end

function Ex2b()
    m = [10, 100, 200, 300, 400]
    n_times = []
    A_times = []
    for m_val in m
        A = alpha_example(1, m_val)
        time_naive = @elapsed naive_hessenberg_red(A)
        time_algo2 = @elapsed Algo_2(A)
        println("m = $m_val")
        push!(n_times, time_naive)
        push!(A_times, time_algo2)
    end
    combined_matrix = hcat(n_times, A_times)
    LLL = latexify(combined_matrix)
    println(LLL)

end

function Ex2c()
    eps1 = [0.4]
    eps = 10 .^ range(log10(1e-1), log10(1e-10), length=10)
    eps1 = append!(eps1, eps)
    eps1 = append!(eps1, 0)
    shift0_vals = []
    shift1_vals = []
    for eps_val in eps1
        A = [3 2; eps_val 1]
        A_S0, U_S0, J_S0 = basic_QR(A, 1, 1e-10, 0)
        println(A_S0[2,1])
        A_S1, U_S1, J_S1 = basic_QR(A, 1, 1e-10, 1)
        println(A_S1[2,1])
        push!(shift0_vals, A_S0[2,1])
        push!(shift1_vals, A_S1[2,1])
    end
    vals = hcat(shift0_vals, shift1_vals)
    L = latexify(vals, env=:table)
    println(L)
end

Ex2c()



function E3_a()
    A= [1 4 4; 3 -1 3; -1 4 4]
    f=z->sin(z)
    F=schur_parlett(A,f)
    LF = latexify(F)
    println(LF)
end

function E3_b(N_max)
    A = rand(100, 100) + 1im * rand(100, 100)
    A = A / norm(A)
    #f = (A, n) -> A^n
    times_schur = []
    times_naive = []
    for N in 1:N_max
        time_schur = @elapsed F = schur_parlett(A, z -> z^N)
        push!(times_schur, time_schur)
        time_naive = @elapsed F = naiveexp(A, N)
        push!(times_naive, time_naive)
    end
    plot_with_moving_average(N_max, times_schur, times_naive, 10)
end

function E4_c()
    eps_values = 10 .^ range(-10, -1, length=100)
    norm_diff = []

    for ε in eps_values
        A = [π 1; 0 π + ε]
        F_exact = [exp(π) (exp(π+ε)-exp(π))/ε;
           0      exp(π+ε)]

        T, Q = schur(Matrix{ComplexF64}(A))
        F_approx = schur_parlett(A, exp)

        push!(norm_diff, norm(F_exact - F_approx))
    end

    plot(
        eps_values, norm_diff, 
        xlabel="ε", ylabel="‖f(A) - F‖", 
        xaxis=:log, yaxis=:log, label="Error vs ε"
    )
    savefig("E4_c.png")
end

function taylor_approx(A::AbstractMatrix, m::Int)
    n = size(A,1)
    TmA = Matrix(I, n, n)
    curr_power = Matrix(I, n, n)
    for k = 1:m
        curr_power = curr_power * A
        TmA = TmA + curr_power/factorial(k)
    end
    return TmA
end

function E5()

    Random.seed!(0)
    n = 256
    A = randn(n,n)
    A = A/norm(A)
    A = A - 3I(n)

    m_values = [1,2,3,4,5,6,7]
    j_values = [0,1,2,3,4,5,6,7]

    error_vals = fill(NaN, length(m_values), length(j_values))
    cpu_times = fill(NaN, length(m_values), length(j_values))

    expA = exp(A)

    for mi in 1:length(m_values)
        m = m_values[mi]
        for ji in 1:length(j_values)
            j = j_values[ji]

            # Measure CPU time for the whole process
            elapsed = @elapsed begin
                # Scale A
                A_scaled = A/(2^j)

                # Compute T_m(A_scaled) using only m-1 multiplications
                TmA = Matrix{Float64}(I, n, n)  # Start with I
                if m >= 1
                    # Add A_scaled directly (no multiplication here)
                    TmA .+= A_scaled
                    curr_power = A_scaled
                    # Compute powers from A^2 to A^m
                    for k in 2:m
                        curr_power = curr_power * A_scaled  # One multiplication
                        TmA .+= curr_power/factorial(k)
                    end
                end

                # Now do scaling-and-squaring: square j times
                for sq in 1:j
                    TmA = TmA * TmA  # each step is one multiplication
                end

                # Compute the error
                err = sqrt(sum(abs2, TmA - expA))
                error_vals[mi, ji] = err

            end

            cpu_times[mi, ji] = elapsed
        end
    end

    # Table 1: Total number of matrix-matrix products = (m-1) + j
    println("Matrix-matrix products count (Table 1):")
    print("       ")
    for j in j_values
        print("j=$j  ")
    end
    println()
    for mi in 1:length(m_values)
        m = m_values[mi]
        print("m=$m: ")
        for j in j_values
            # total multiplications: (m - 1) for T_m(A) + j for squaring
            print((m-1) + j, "    ")
        end
        println()
    end

    # Table 2: log10 error
    println("\nlog10 error (Table 2):")
    log10_error = log10.(error_vals)
    # Print with formatting
    print("       ")
    for j in j_values
        print("j=$j       ")
    end
    println()
    for mi in 1:length(m_values)
        m = m_values[mi]
        print("m=$m: ")
        for ji in 1:length(j_values)
            val = log10_error[mi, ji]
            isfinite(val) ? print(round(val, digits=4), "    ") : print("NaN     ")
        end
        println()
    end

    # Table 3: CPU-time
    println("\nCPU time (Table 3):")
    print("       ")
    for j in j_values
        print("j=$j       ")
    end
    println()
    for mi in 1:length(m_values)
        m = m_values[mi]
        print("m=$m: ")
        for ji in 1:length(j_values)
            val = cpu_times[mi, ji]
            print(round(val, digits=6), "  ")
        end
        println()
    end
end


#E3_a()
# E3_b(300)
# E3_b(300) Ex2b()
E4_c()
# E5()