using StateSpace
using Distributions
using Base.Test

println("Testing Kalman Filter...")

function random_cov(n)
	A = rand(n, n)
	A = A + A'
	return A + n * eye(n)
end

srand(0)

F = diagm(ones(4)) * 0.9
V = random_cov(4)
G = [1.0 0.0 0.5 0.1;
	 0.0 0.5 1.0 1.0]
W = random_cov(2)

m = LinearGaussianSSM(F, V, G, W)
x0 = MvNormal(randn(4), diagm(ones(4) * 100.0))

x1 = predict(m, x0)
# println(mean(x1))

y1 = m.G(1) * mean(x1) + randn(2) / 10
u1 = update(m, x1, y1)
# println(mean(u1))

xx, yy = simulate(m, 100, x0)
yy[1, 50] = NaN  # throw in a missing value

# using PyPlot
# plot(xx', "k")
# plot(yy', "r")
# readline()

fs = filter(m, yy, x0)
# print(fs)

# plot(xx', "k")
# plot(mean(fs)', "r")
# readline()

y_new = fs.observations[:, end] + randn(2) / 10
update!(m, fs, y_new)

ss = smooth(m, fs)


@assert loglikelihood(fs) < loglikelihood(ss)


println("KalmanFilter.jl passed.\n")