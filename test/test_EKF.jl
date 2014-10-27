using StateSpace

println("Testing ExtendedKalmanFilter.jl...")


f(x) = [0.2 * x[1]^2, 0.1 * x[2]^2]
m = 2
V = diagm([0.3, 0.1])

g(x) = [x[1], x[2]^2, x[1] * x[2]]
n = 3
W = diagm(ones(3) * 0.1)


mod = NonlinearGaussianSSM(f, m, V, g, n, W)
x0 = MvNormal([1.0, 1.0], [100.0 0.0; 0.0 100.0])

x1 = predict(mod, x0)
# println(mean(x1))

y1 = observe(mod, x1)
u1 = update(mod, x1, mean(y1))
# println(mean(u1))

xx, yy = simulate(mod, 100, x0)
yy[1, 50] = NaN  

# using PyPlot
# plot(xx', "k")
# plot(yy', "r")
# readline()

fs = filter(yy, mod, x0)

# plot(xx', "k")
# plot(mean(fs)', "r")
# readline()

y_new = fs.observations[:, end] + randn(mod.n) / 10
update!(mod, fs, y_new)


println("ExtendedKalmanFilter.jl passed.\n")