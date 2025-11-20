using MAT
using Images
using ImageView
using Statistics
using LinearAlgebra

#=
(a). Load the data and display a face
=#
file = matread("olivetti_faces.mat")
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

