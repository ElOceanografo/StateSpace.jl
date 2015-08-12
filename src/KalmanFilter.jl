using Distributions

issquare(x::Matrix) = size(x, 1) == size(x, 2)

type LinearGaussianSSM{T} <: AbstractStateSpaceModel
	F::Matrix{T}
	V::Matrix{T}
	G::Matrix{T}
	W::Matrix{T}

	function LinearGaussianSSM(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, W::Matrix{T})
		@assert issquare(F)
		@assert issquare(V)
		@assert issquare(W)
		@assert size(F) == size(V)
		@assert size(G, 1) == size(W, 1)
		return new(F, V, G, W)
	end
end

function LinearGaussianSSM{T <: Real}(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, 
		W::Matrix{T})
	return LinearGaussianSSM{T}(F, V, G, W)
end

# Univariate state and data
function LinearGaussianSSM{T<:Real}(F::T, V::T, G::T, W::T)
	f = Array(T, 1, 1); f[1] = F
	v = Array(T, 1, 1); v[1] = V
	g = Array(T, 1, 1); g[1] = G
	w = Array(T, 1, 1); w[1] = W
	return LinearGaussianSSM(f, v, g, w)
end

# Univariate state, n-d data
function LinearGaussianSSM{T<:Real}(F::T, V::T, G::Matrix{T}, W::Matrix{T})
	f = Array(T, 1, 1); f[1] = F
	v = Array(T, 1, 1); v[1] = V
	return LinearGaussianSSM(f, v, G, W)
end

# m-d state, univariate data
function LinearGaussianSSM{T<:Real}(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, W::T)
	w = Array(T, 1, 1); w[1] = W
	return LinearGaussianSSM(F, V, G, w)
end

## Core methods
function predict(m::LinearGaussianSSM, x::AbstractMvNormal)
	return MvNormal(m.F * mean(x), m.F * cov(x) * m.F' + m.V)
end

function observe(m::LinearGaussianSSM, x::AbstractMvNormal)
	return MvNormal(m.G * mean(x), m.G * cov(x) * m.G' + m.W)
end

function update(m::LinearGaussianSSM, pred::AbstractMvNormal, y)
	innovation = y - m.G * mean(pred)
	innovation_cov = m.G * cov(pred) * m.G' + m.W
	K = cov(pred) * m.G' * inv(innovation_cov)
	mean_update = mean(pred) + K * innovation
	cov_update = (eye(cov(pred)) - K * m.G) * cov(pred)
	return MvNormal(mean_update, cov_update)
end

function update!(m::LinearGaussianSSM, fs::FilteredState, y)
	x_pred = predict(m, fs.state[end])
	x_filt = update(m, x_pred, y)
	push!(fs.state, x_filt)
	fs.observations = [fs.observations y]
	return fs
end


function filter{T}(y::Array{T}, m::LinearGaussianSSM{T}, x0::AbstractMvNormal)
	x_filtered = Array(AbstractMvNormal, size(y, 2))
	loglik = 0.0
	x_pred = predict(m, x0)
	x_filtered[1] = update(m, x_pred, y[:, 1])
	loglik = logpdf(x_filtered[1], mean(x_pred)) + 
		logpdf(observe(m, x_filtered[1]), y[:,1])
	for i in 2:size(y, 2)
		x_pred = predict(m, x_filtered[i-1])
		# Check for missing values in observation
		if any(isnan(y[:, i]))
			x_filtered[i] = x_pred
		else
			x_filtered[i] = update(m, x_pred, y[:, i])
			loglik += logpdf(observe(m, x_filtered[i]), y[:, i])
		end
		loglik += logpdf(x_pred, mean(x_filtered[i]))
	end
	return FilteredState(y, x_filtered, loglik)
end

function smooth{T}(m::LinearGaussianSSM{T}, fs::FilteredState{T})
	n = size(fs.observations, 2)
	smooth_dist = Array(AbstractMvNormal, n)
	smooth_dist[end] = fs.state[end]
	loglik = logpdf(observe(m, smooth_dist[end]), fs.observations[:, end])
	for i in (n - 1):-1:1
		state_pred = predict(m, fs.state[i])
		P = cov(fs.state[i])
		J = P * m.F' * inv(cov(state_pred))
		x_smooth = mean(fs.state[i]) + J * 
			(mean(smooth_dist[i+1]) - mean(state_pred))
		P_smooth = P + J * (cov(smooth_dist[i+1]) - cov(state_pred)) * J'
		smooth_dist[i] = MvNormal(x_smooth, P_smooth)
		loglik += logpdf(predict(m, smooth_dist[i]), mean(smooth_dist[i+1]))
		if ! any(isnan(fs.observations[:, i]))
			loglik += logpdf(observe(m, smooth_dist[i]), fs.observations[:, i])
		end
	end
	return SmoothedState(fs.observations, smooth_dist, loglik)
end


function simulate{T}(m::LinearGaussianSSM{T}, n::Int64, x0::AbstractMvNormal)
	x = zeros(T, length(x0), n)
	y = zeros(T, size(m.G, 1), n)
	x[:, 1] = rand(MvNormal(m.F * mean(x0), m.V))
	y[:, 1] = rand(MvNormal(m.G * x[:, 1], m.W))
	for i in 2:n
		x[:, i] = rand(MvNormal(m.F * x[:, i-1], m.V))
		y[:, i] = rand(MvNormal(m.G * x[:, i], m.W))
	end
	return (x, y)
end

