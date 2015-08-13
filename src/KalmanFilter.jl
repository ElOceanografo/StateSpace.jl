using Distributions

issquare(x::Matrix) = size(x, 1) == size(x, 2)

type LinearGaussianSSM{T} <: AbstractGaussianSSM
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
process_matrix(m::LinearGaussianSSM, state) = m.F
observation_matrix(m::LinearGaussianSSM, state) = m.G


