import StatsBase

"""
Create a nonlinear state space model.

#### Parameters
- f : Function. Process function, maps (x_{t}, t+1) -> x_{t+1}.
- b : Funciton. (Optional) input function, mapping (x_{t}, u_{t}, t) -> x_{t}.
- obs_loglik : Function. Should take a state, observation, and time, and
	return a log-likelihood value, i.e. log(P(y | x, t))

#### Returns
- NonlinearSSM object.
"""
type NonlinearSSM
	f::Function
	b::Function
	obs_loglik::Function
	nx::Int
	nu::Int
	ny::Int
end

function NonlinearSSM(f::Function, obs_loglik::Function,
		nx::Int, ny::Int, nu::Int)
	b(x::Vector, u::Vector, t::Int) = x
	NonlinearSSM(f, b, obs_loglik, nx, ny, ny)
end

"""
Type representing a particle filter.

#### Parameters
- nx : Int. Size of the state vector.
- nparticles : Int.  Number of particles in the ensemble.
- jitter : Function.  Returns a vector of nx random numbers.
"""
type ParticleFilter
	nx::Int
	nparticles::Int
	jitter::Function
end

function jitter(filter::ParticleFilter)
	noise = zeros(filter.nx, filter.nparticles)
	for i in 1:filter.nparticles
		noise[:, i] = filter.jitter()
	end
	return noise
end


function predict(m::NonlinearSSM, x::Vector; u::Vector=zeros(m.nu), t::Int=1)
	x = m.f(x, t)
	x = m.b(x, u, t)
	return x
end

function predict(m::NonlinearSSM, ensemble::Matrix; u::Vector=zeros(m.nu),
		t::Int=1)
	output = similar(ensemble)
	for i in 1:size(ensemble, 2)
		output[:, i] = m.b(m.f(ensemble[:, i], t), u, t)
	end
	return output
end

# do prediction in place, modifying ensemble
function predict!(m::NonlinearSSM, ensemble::Matrix; u::Vector=zeros(m.nu),
		t::Int=1)
	for i in 1:size(ensemble, 2)
		ensemble[:, i] = m.f(ensemble[:, i], u, t)
	end
end


function simulate(m::NonlinearSSM, nt::Int, x0::Vector)
	x = zeros(m.nx, nt)
	x[:, 1] = predict(m, x0)
	for i in 2:nt
		x[:, i] = predict(m, x[:, i-1])
	end
	return x
end

function observe(m::NonlinearSSM, x::Vector, y::Vector, t::Int=1)
	return m.obs_loglik(x, y, t)
end

function observe(m::NonlinearSSM, ensemble::Matrix, y::Vector, t::Int=1)
	return Float64[observe(m, ensemble[:, i], y, t) for i in 1:size(ensemble, 2)]
end

function update(m::NonlinearSSM, pred::Matrix, y::Vector, filter::ParticleFilter,
		t::Int=1)
	n = filter.nparticles
	likelihoods = exp.(observe(m, pred, y, t))
	weights = StatsBase.Weights(likelihoods / sum(likelihoods))
	indices = sample(1:n, weights, n)
	return pred[:, indices] + jitter(filter)
end

# update the the ensemble in place
function update!(m::NonlinearSSM, pred::Matrix, y::Vector, filter::ParticleFilter,
		t::Int=1)
	n = size(pred, 2)
	likelihoods = exp(observe(m, pred, y, t))
	weights = StatsBase.WeightVec(likelihoods / sum(likelihoods))
	indices = sample(1:n, weights, n)
	pred = pred[:, indices] + jitter(filter)
end


function filter(m::NonlinearSSM, y::Matrix, x0::Matrix,
		f::ParticleFilter, u::Matrix=zeros(m.nu, size(y, 2)))
	@assert size(x0, 1) == m.nx == f.nx
	@assert size(x0, 2) == f.nparticles
	@assert size(u, 1) == m.nu
	@assert size(y, 1) == m.ny
	@assert size(u, 2) == size(y, 2)

	nt = size(y, 2)
	fs = zeros(f.nx, f.nparticles, nt)
	x_pred = predict(m, x0, u=u[:, 1], t=1)
	fs[:, :, 1] = update(m, x0, y[:, 1], f, 1)
	for i in 2:nt
		x_pred = predict(m, fs[:, :, i-1], u=u[:, i], t=i)
		fs[:, :, i] = update(m, x_pred, y[:, i], f, i)
	end
	return fs
end
