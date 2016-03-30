"""
Forecast the state of the process at the next time step.

#### Parameters
- m : AbstractGaussianSSM.  Model of the state evolution and observation processes.
- x : AbstractMvNormal.  Current state estimate.

#### Returns
- MvNormal distribution representing the forecast and its associated uncertainty.
"""
function predict(m::AbstractGaussianSSM, x::AbstractMvNormal;
		u::Array=zeros(m.nu), t::Real=0.0)
    F = process_matrix(m, x, t)
    CI = control_input(m, u, t)
    return MvNormal(F * mean(x) + CI, F * cov(x) * F' + m.V(t))
end


"""
Observe the state process with uncertainty.

#### Parameters
- m : AbstractGaussianSSM.  Model of the state evolution and observation processes.
- x : AbstractMvNormal.  Current state estimate.

#### Returns
- MvNormal distribution, representing the probability of recording any
particular observation data.
"""
function observe(m::AbstractGaussianSSM, x::AbstractMvNormal, t::Real=0.0)
	G = observation_matrix(m, x, t)
	return MvNormal(G * mean(x), G * cov(x) * G' + m.W(t))
end

"""
Given a process/observation model and a set of data, estimate the states of the
hidden process at each time step.

#### Parameters
- m : AbstractGaussianSSM.  Model of the state evolution and observation processes.
- y : Array of data.  Each column is a set of observations at one time, while
each row is one observed variable.  Can contain NaNs to represent missing data.
- x0 : AbstractMvNorm. Multivariate normal distribution representing our estimate
of the hidden state and our uncertainty about that estimate before our first observation.
(i.e., if the first data arrives at t=1, this is where we think the process is at t=0.)

#### Returns
- A FilteredState object holding the estimates of the hidden state.

#### Notes
This function only does forward-pass filtering--that is, the state estimate at time
t incorporates data from 1:t, but not from t+1:T.  For full forward-and-backward
filtering, run `smooth` on the FilteredState produced by this function.
"""
function _filter{T}(m::AbstractGaussianSSM, y::Array{T}, x0::AbstractMvNormal,
		u::Array{T}, times::Vector{T}, filter::AbstractKalmanFilter)
	x_filtered = Array(AbstractMvNormal, size(y, 2))
	loglik = 0.0
	x_pred = predict(m, x0, u=u[:, 1], t=times[1])
	x_filtered[1] = update(m, x_pred, y[:, 1], filter, 1)
	loglik = logpdf(x_filtered[1], mean(x_pred))
	if ! any(isnan(y[:, 1]))
		loglik += logpdf(observe(m, x_filtered[1]), y[:,1])
	end
	for i in 2:size(y, 2)
		x_pred = predict(m, x_filtered[i-1], u=u[:, i], t=times[i])
		# Check for missing values in observation
		if any(isnan(y[:, i]))
			x_filtered[i] = x_pred
		else
			x_filtered[i] = update(m, x_pred, y[:, i], filter, i)
			loglik += logpdf(observe(m, x_filtered[i]), y[:, i])
		end
		loglik += logpdf(x_pred, mean(x_filtered[i]))
	end
	return FilteredState(y, x_filtered, u, times, loglik, false)
end


function filter{T}(m::LinearGaussianSSM, y::Array{T}, x0::AbstractMvNormal;
		u::Matrix{T}=zeros(m.nu, size(y, 2)), filter::LinearKalmanFilter=KF(),
		times::Vector{T}=zeros(size(y, 2)))
	return _filter(m, y, x0, u, times, filter)
end

function filter{T}(m::NonlinearGaussianSSM, y::Array{T}, x0::AbstractMvNormal;
		u::Matrix{T}=zeros(m.nu, size(y, 2)), filter::NonlinearKalmanFilter=EKF(),
		times::Vector{T}=zeros(size(y, 2)))
	return _filter(m, y, x0, u, times, filter)
end


"""
Given a process/observation model and a set of data, estimate the states of the
hidden process at each time step using forward and backward passes.

This function has two methods.  It can take either an AbstractGaussianSSM and a the
FilteredState produced by a previous call to `filter`, or it can take a model, data,
set, and initial state estimate, in which case it does the forward and backward
passes at once.

"""
function _smooth{T}(m::AbstractGaussianSSM, fs::FilteredState{T})
	n = size(fs.observations, 2)
	smooth_dist = Array(AbstractMvNormal, n)
	smooth_dist[end] = fs.state[end]
	if ! any(isnan(fs.observations[:, 1]))
		loglik = logpdf(observe(m, smooth_dist[end], fs.times[end]), 
			fs.observations[:, end])
	end
	for i in (n - 1):-1:1
		state_pred = predict(m, fs.state[i], u=fs.input[:, i], t=fs.times[i])
		P = cov(fs.state[i])
		F = process_matrix(m, fs.state[i], fs.times[i])
		J = P * F' * inv(cov(state_pred))
		x_smooth = mean(fs.state[i]) + J *
			(mean(smooth_dist[i+1]) - mean(state_pred))
		P_smooth = P + J * (cov(smooth_dist[i+1]) - cov(state_pred)) * J'
		smooth_dist[i] = MvNormal(x_smooth, P_smooth)
		loglik += logpdf(predict(m, smooth_dist[i], u=fs.input[:, i], t=fs.times[i]), 
			mean(smooth_dist[i+1]))
		if ! any(isnan(fs.observations[:, i]))
			loglik += logpdf(observe(m, smooth_dist[i], fs.times[i]), fs.observations[:, i])
		end
	end
	return FilteredState(fs.observations, smooth_dist, fs.input, fs.times, loglik, true)
end


function smooth{T}(m::AbstractGaussianSSM, fs::FilteredState{T})
	return _smooth(m, fs)
end

## Linear Gaussian method
function smooth{T}(m::LinearGaussianSSM, y::Array{T}, x0::AbstractMvNormal;
		u::Matrix{T}=zeros(m.nu, size(y, 2)), filter::LinearKalmanFilter=KF())
	fs = filter(m, y, x0, u=u, filter=filter)
	return _smooth(m, fs)
end

## Nonlinear Gaussian method
function smooth{T}(m::NonlinearGaussianSSM, y::Array{T}, x0::AbstractMvNormal;
		u::Matrix{T}=zeros(m.nu, size(y, 2)), filter::NonlinearKalmanFilter=EKF())
	fs = filter(m, y, x0, u=u, filter=filter)
	return _smooth(m, fs)
end


"""
Simulate a realization of a state-space process and observations of it.

#### Parameters
- m : AbstractGaussianSSM.  Model of the state evolution and observation processes.
- x0 : AbstractMvNorm. Distribution of the state process's starting value.

#### Returns
- (x, y) : Tuple of the process and observation arrays.
"""
function simulate(m::AbstractGaussianSSM, n::Int64, x0::AbstractMvNormal)
	F = process_matrix(m, x0, 1)
	G = observation_matrix(m, x0, 1)
	x = zeros(size(F, 1), n)
	y = zeros(size(G, 1), n)
	x[:, 1] = rand(MvNormal(F * mean(x0), m.V(1)))
	y[:, 1] = rand(MvNormal(G * x[:, 1], m.W(1)))
	for i in 2:n
		F = process_matrix(m, x[:, i-1], i)
		G = observation_matrix(m, x[:, i-1], i)
		x[:, i] = F * x[:, i-1] + rand(MvNormal(m.V(i)))
		y[:, i] = G * x[:, i] + rand(MvNormal(m.W(i)))
	end
	return (x, y)
end


"""
Fit a state-space model to data using maximum likelihood.


"""
function fit(build_func, y, x0, params0, args...)

	function objective(params)
		ssm = build_func(params)
		return -loglikelihood(smooth(m, y, x0))
	end
	fit = optimize(objective, params0, args...)
end
