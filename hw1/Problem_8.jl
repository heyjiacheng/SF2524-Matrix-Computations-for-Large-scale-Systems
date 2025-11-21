using MAT
using Images
using ImageView
using Statistics
using LinearAlgebra
using Plots
using FileIO

function normalize_image(img)
    return (img .- minimum(img)) ./ maximum(img .- minimum(img))
end

#=
(a). Load the data and display a face
=#
file = matread("hw1/olivetti_faces.mat")
faces = file["faces"]

face_88 = faces[:, :, 88]
# imshow(face_88)


#=
(b). Construct the data matrix X and X_c. Compare the computation times
=#
X = reshape(faces, 64*64, 400)

# compute the average face
x_mean = mean(X, dims = 2)
X_c = X .- x_mean

# Compare the computation times
B = [rand(size(X_c, 1)) for _ in 1:100]

# warm-up:
C = X_c * X_c'
for b in B
    C * b
end
for b in B
    X_c * (X_c' * b)
end

println(size(X_c))
println(size(C))

time_1 = @elapsed C = X_c * X_c'
time_2 = @elapsed begin
    for b in B
        C * b
    end
end
time_3 = @elapsed begin
    for b in B
        X_c * (X_c' * b)
    end
end

println("Time to form C: $time_1 seconds")
println("Time for C * b: $time_2 seconds")
println("Time for X_c * (X_c' * b): $time_3 seconds")

#=
Power Method to find the largest eigenpair of C
=#
function rayleigh_quotient(A, v)
    return dot(v, A * v) / dot(v, v)
end

function mulC(v)
    m = size(X_c, 2)
    return (X_c * (X_c' * v)) / (m - 1)
end

function power_method_A(A, v0; iter = 30)
    v = v0 / norm(v0)
    eigenvalue_exact = maximum(abs, eigvals(A))
    eigenvalue_error = Float64[]

    eigenvalue_old = rayleigh_quotient(A, v)
    v_new = A * v
    v_new /= norm(v_new)
    eigenvalue_new = rayleigh_quotient(A, v_new)

    i = 0
    while i <= iter
        eigenvalue_old = eigenvalue_new
        push!(eigenvalue_error, abs(eigenvalue_new - eigenvalue_exact) / abs(eigenvalue_exact))

        v = v_new
        v_new = A * v
        v_new /= norm(v_new)
        eigenvalue_new = rayleigh_quotient(A, v_new)
        i += 1
    end

    println("\nPower Method after $iter iterations:")
    println("  Largest eigenvalue: ", rayleigh_quotient(A, v_new))

    return eigenvalue_error, v_new, eigenvalue_new
end

function power_method_B(B, v0, lambda_exact; iter=30)
    error = Float64[]
    v = v0 / norm(v0)
    lambda_new = 0.0
    v_new = zeros(size(v))

    for i in 1:iter
        v_new = mulC(v)

        lambda_new = dot(v, v_new)
        push!(error, abs(lambda_new - lambda_exact) / abs(lambda_exact))
        
        v = v_new / norm(v_new)
    end

    println("\nPower Method after $iter iterations:")
    println("Largest eigenvalue: ", lambda_new)

    return error, v_new, lambda_new
end

v_0 = ones(size(X_c, 1))
C = (X_c * X_c') / (size(X_c, 2) - 1)
lambda_exact = maximum(abs, eigvals(C))

error_1, v_1, λ_1 = power_method_A(C, v_0; iter=30)
error_2, v_2, λ_2 = power_method_B(X_c, v_0, lambda_exact; iter=30)

time_pm_A = @elapsed error_1, v_1, λ_1 = power_method_A(C, v_0; iter=30)
time_pm_B = @elapsed error_2, v_2, λ_2 = power_method_B(X_c, v_0, lambda_exact; iter=30)

println("\nTime for Power Method on C: $time_pm_A seconds")
println("Time for Power Method on X_c: $time_pm_B seconds") 

# Display the eigenface corresponding to the largest eigenvalue
eigenface_1 = reshape(v_1, 64, 64)
save("eigenface1.png", normalize_image(eigenface_1))

# #=
# Use the Arnoldi method to compute the five largest eigenvalues and corresponding eigenvectors of C
# =#
# function arnoldi(Xc, v0; m=35)
#     n = size(Xc, 1)
#     V = zeros(Float64, n, m + 1)
#     H = zeros(Float64, m + 1, m)
#     V[:, 1] = v0 / norm(v0)

#     mcols = size(Xc, 2)
#     mulC_local(v) = (Xc * (Xc' * v)) / (mcols - 1)

#     for k in 1:m
#         w = mulC_local(view(V, :, k))

#         for j in 1:k
#             H[j, k] = dot(view(V, :, j), w)
#             w -= H[j, k] * view(V, :, j)
#         end

#         H[k + 1, k] = norm(w)
#         if H[k + 1, k] == 0
#             println("Arnoldi terminated early at step $k")
#             return V[:, 1:k], H[1:k, 1:k-1]
#         end
        
#         V[:, k + 1] = w / H[k + 1, k]
#     end

#     return V, H
# end

# function arnoldi_residuals(Xc, v0; msteps=35, num=5)
#     n = size(Xc, 1)
#     V, H = arnoldi(Xc, v0; m=msteps)

#     kmax = size(H, 2)

#     residuals = zeros(Float64, num, msteps)

#     m = size(Xc, 2)

#     for k in 1:kmax
#         H_k = H[1:k, 1:k]
#         eigvals_H, eigvecs_H = eigen(H_k)
#         idx = sortperm(abs.(eigvals_H), rev=true)
#         λ_sorted = eigvals_H[idx]
#         y_sorted = eigvecs_H[:, idx]

#         mulC_local(v) = (Xc * (Xc' * v)) / (m - 1)

#         num_k = min(num, k)

#         for j in 1:num_k
#             λ_j = λ_sorted[j]
#             y_j = y_sorted[:, j]

#             Vk = V[:, 1:k]
#             x = Vk * y_j

#             Cx = mulC_local(x)
#             r = norm(Cx - λ_j * x) / norm(Cx)

#             residuals[j, k] = r
#         end
#     end

#     return residuals, V, H
# end

# msteps = 35
# num = 5
# v_0 = rand(size(X_c, 1))
# residuals, V, H = arnoldi_residuals(X_c, v_0; msteps=msteps, num=num)
# println("Residuals matrix:")
# println(residuals)

# # Visualize the semilog plot of the residuals
# iters = 1:msteps
# labels = ["λ₁", "λ₂", "λ₃", "λ₄", "λ₅"]

# plt = plot(;
#     xlabel="Iteration",
#     ylabel="Relative Residuals",
#     yscale=:log10,
#     yrange=(1e-16, 1e0),
#     title="Arnoldi Method Convergence"
# )


# for j in 1:num
#     plot!(plt, iters[1:msteps], residuals[j, :], label=labels[j])
# end
# savefig(plt, "arnoldi_convergence.png")


# # Display the eigenfaces corresponding to the five largest eigenvalues
# kfinal = size(H, 2)
# Hk_final = H[1:kfinal, 1:kfinal]
# eig_final = eigen(Hk_final)
# λs_final = eig_final.values
# Ys_final = eig_final.vectors

# idx = sortperm(abs.(λs_final), rev=true)[1:num]
# λs_top = λs_final[idx]
# Ys_top = Ys_final[:, idx]

# Vk_final = V[:, 1:kfinal]

# for j in 1:5
#     y_j = Ys_top[:, j]
#     eigenvector_j = Vk_final * y_j
#     eigenface_j = reshape(eigenvector_j, 64, 64)
#     save("eigenface$(j).png", normalize_image(eigenface_j))
# end

# #=
# The spookist eigenface in the first 20 eigenfaces.
# =#
# msteps = 35
# num = 20
# residuals, V, H = arnoldi_residuals(X_c, v_0; msteps=msteps, num=num)
# kfinal = size(H, 2)
# Hk_final = H[1:kfinal, 1:kfinal]
# eig_final = eigen(Hk_final)

# λs = eig_final.values
# Ys = eig_final.vectors

# idx = sortperm(abs.(λs), rev=true)[1:num]

# Vk_final = V[:, 1:kfinal]

# for j in 1:num
#     y_j = Ys[:, idx[j]]
#     eigenface_vec = Vk_final * y_j
#     eigenface_img = reshape(eigenface_vec, 64, 64)
#     save("arnoldi_eigenface$(j).png", normalize_image(eigenface_img))
#     save("arnoldi_eigenface_neg$(j).png", normalize_image(-eigenface_img))
# end