using StateSpace

println("Testing ExtendedKalmanFilter.jl...")

# Process
f(x) = [x[1] + x[2], x[1]^2, 3 * x[3]^4]
m = 3
V = diagm(ones(3))
# Observation
g(x) = [x[1] * x[2] * x[3], sum(x)]
n = 2
W = diagm(ones(2) * 0.2)

m = NonlinearGaussianSSM(f, m, V, g, n, W)


println("ExtendedKalmanFilter.jl passed.\n")