println("Testing particle filter...")
# Bearing-only tracking example from Gordon et al. 1993
Phi = eye(4)
Phi[1, 2] = Phi[3, 4] = 1
Gamma = [0.5 0; 1 0; 0 0.5; 0 1]
q = 0.001
r = 0.005
x0 = [-0.05, 0.01, 0.7, -0.055]

process(x, t) = Phi * x + Gamma * q * randn(2)
bearing(x) = atan(x[2] / x[1])
obs_loglik(x, y, t) = logpdf(Normal(bearing(x), r), y[1])


mod = NonlinearSSM(process, obs_loglik, 4, 1, 1)
filt = ParticleFilter(4, 1000, () -> zeros(4))

nt = 24
srand(1234)
xx = simulate(mod, nt, x0)
yy = (bearing(xx) + randn(nt) * r)'

# using PyPlot
# subplot(111, aspect="equal")
# plot(xx[1, :]', xx[3, :]')
# scatter(0, 0)


x0_prior = MvNormal([0, 0, 0.4, -0.05], diagm([0.5, 0.005, 0.3, 0.01].^2))
ensemble_0 = rand(x0_prior, filt.nparticles)

ensemble = predict(mod, ensemble_0)

observe(mod, ensemble[:, 1], yy[:, 1])
observe(mod, ensemble, yy[:, 1])

ensemble_new = update(mod, ensemble, yy[:, 1], filt)

# scatter(xx[1,1], xx[3,1])
# plot(ensemble[1,:]', ensemble[3, :]', "k+")
# plot(ensemble_new[1,:]', ensemble_new[3, :]', "r.")

fs = filter(mod, yy, ensemble_0, filt)

fs_mean = mean(fs, 2)
# plot(fs_mean[1,:]', fs_mean[3,:]')


println("Particle filter passed.")