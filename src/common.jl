using Distributions
import Distributions: mean, var, cov, rand

abstract AbstractStateSpaceModel
typealias AbstractSSM AbstractStateSpaceModel

type FilteredState{T, D<:ContinuousMultivariateDistribution}
	observations::Array{T, 2}
	state_dist::Array{D}
	loglik::T
end

function show{T}(io::IO, fs::FilteredState{T})
	n = length(fs.state_dist)
	dobs = size(fs.observations, 1)
	dstate = length(fs.state_dist[1])
	println("FilteredState{$T}")
	println("$n estimates of $dstate-D process from $dobs-D observations")
	println("Log-likelihood: $(fs.loglik)")
end

for op in (:mean, :var, :cov, :cor, :rand)
	@eval begin
		function ($op){T}(s::FilteredState{T})
			result = Array(T, length(s.state_dist[1]), length(s.state_dist))
			for i in 1:length(s.state_dist)
				result[:, i] = ($op)(s.state_dist[i])
			end
			return result
		end
	end
end

