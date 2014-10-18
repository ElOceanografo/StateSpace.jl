using Distributions

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

function mean{T}(s::FilteredState{T})
	mu = Array(T, length(s.state_dist[1]), length(s.state_dist))
	for i in 1:length(s.state_dist)
		mu[:, i] = mean(s.state_dist[i])
	end
	return mu
end