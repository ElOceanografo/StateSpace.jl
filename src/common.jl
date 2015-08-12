using Distributions
import Distributions: mean, var, cov, rand

abstract AbstractStateSpaceModel
typealias AbstractSSM AbstractStateSpaceModel

type FilteredState{T, D<:ContinuousMultivariateDistribution}
	observations::Array{T, 2}
	state::Array{D}
	loglik::T
end

typealias SmoothedState FilteredState

function show{T}(io::IO, fs::FilteredState{T})
	n = length(fs.state)
	dobs = size(fs.observations, 1)
	dstate = length(fs.state[1])
	println("FilteredState{$T}")
	println("$n estimates of $dstate-D process from $dobs-D observations")
	println("Log-likelihood: $(fs.loglik)")
end

function show{T}(io::IO, fs::SmoothedState{T})
	n = length(fs.state)
	dobs = size(fs.observations, 1)
	dstate = length(fs.state[1])
	println("SmoothedState{$T}")
	println("$n estimates of $dstate-D process from $dobs-D observations")
	println("Log-likelihood: $(fs.loglik)")
end

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

