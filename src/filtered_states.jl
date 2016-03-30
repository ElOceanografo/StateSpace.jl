"""
Data structure representing the estimated state of a state-space model
after filtering/smoothing.

#### Fields
- observations : Array of observations. Each column is  a single observation vector.
- state : Array of (possibly multivariate) Distributions. Each distribution
represents the estimate and uncertainty of the hidden state at that time.
- loglik : The log-likelihood of the model, i.e. the probabilty of the observations
given the model and state estimates.
- smoothed : Bool. Whether the state has been smoothed, or just filtered.
"""
type FilteredState{T, D<:ContinuousMultivariateDistribution}
	observations::Array{T, 2}
	state::Array{D}
	input::Array{T, 2}
	times::Vector{T}
	loglik::T
	smoothed::Bool
end


function show{T}(io::IO, fs::FilteredState{T})
	n = length(fs.state)
	dobs = size(fs.observations, 1)
	dstate = length(fs.state[1])
	s = ifelse(fs.smoothed, "smoothed", "filtered")
	println("FilteredState{$T}")
	println("$n $s estimates of $dstate-D process from $dobs-D observations")
	println("Log-likelihood: $(fs.loglik)")
end


"""
Returns the log-likelihood of the FilteredState object.
"""
loglikelihood(fs::FilteredState) = fs.loglik

for op in (:mean, :var, :cov, :cor, :rand)
	@eval begin
		function ($op){T}(s::FilteredState{T})
			result = Array(T, length(s.state[1]), length(s.state))
			for i in 1:length(s.state)
				result[:, i] = ($op)(s.state[i])
			end
			return result
		end
	end
end
