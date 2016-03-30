using StateSpace

println("Testing Ensemble Kalman Filter...")

# Define process model: Ricker growth model, with two populations
r1 = 0.03 							# 10% intrinsic growth rate for pop 1
r2 = 0.06 							# 14% intrinsic growth rate for pop 1
K1 = 20000 							# carrying capacity for pop 1
K2 = 10000 							# carrying capacity for pop 2

# process model
function f(x)
	[x[1] .* exp(r1 * (1 - (x[1] ./ K1))), 
	 x[2] .* exp(r2 * (1 - (x[2] ./ K2)))]
end

V = diagm([3000.0, 5000.0]) 		# covariance of (additive) process noise

# Define observation model
g(x) = 0.01 .* x
W = diagm([100.0, 250.0])

mod = NonlinearGaussianSSM(f, V, g, W)
x0 = MvNormal([1000.0, 500.0], [100.0 0.0; 0.0 100.0])

x1 = predict(mod, x0)
# println(mean(x1))

y1 = observe(mod, x1)
u1 = update(mod, x1, mean(y1))
# println(mean(u1))

xx, yy = simulate(mod, 200, x0)
yy[1, 50] = NaN  


filt = EnKF()
fs = filter(mod, yy, x0, filt)
fs_ensemble = filter(mod, yy, x0, filt, return_ensemble=true)
@assert size(fs_ensemble) == (mod.nx, filt.nparticles, size(yy, 2))
y_new = fs.observations[:, end] + randn(size(mod.V(1), 1)) / 10
update!(mod, fs, y_new)

ss = smooth(mod, fs)
@assert loglikelihood(fs) < loglikelihood(ss)


println("Extended Kalman Filter passed.\n")