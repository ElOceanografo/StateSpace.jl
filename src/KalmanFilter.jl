using Distributions

issquare(x::Matrix) = size(x, 1) == size(x, 2)

#####################################################
# DEFINE TYPES
#####################################################
type LinearGaussianSSM{T} <: AbstractLinearGaussian
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

#Linear Gaussian State space model with control input
type LinearGaussianCISSM{T} <: AbstractLinearGaussian
	F::Matrix{T}
	V::Matrix{T}
	G::Matrix{T}
	W::Matrix{T}
    B::Matrix{T}
    u::Vector{T}

	function LinearGaussianCISSM(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, W::Matrix{T}, B::Matrix{T}, u::Vector{T})
		@assert issquare(F)
		@assert issquare(V)
		@assert issquare(W)
        @assert issquare(B)
		@assert size(F) == size(V)
		@assert size(G, 1) == size(W, 1)
        @assert size(B,2) == size(u,1)
		return new(F, V, G, W, B, u)
	end
end

#####################################################
# DEFINE CONSTRUCTOR METHODS
#####################################################

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

function LinearGaussianCISSM{T <: Real}(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, W::Matrix{T}, B::Matrix{T}, u::Vector{T})
	return LinearGaussianCISSM{T}(F, V, G, W, B, u)
end

# Univariate state and data
function LinearGaussianCISSM{T<:Real}(F::T, V::T, G::T, W::T, B::T, U::T)
	f = Array(T, 1, 1); f[1] = F
	v = Array(T, 1, 1); v[1] = V
	g = Array(T, 1, 1); g[1] = G
	w = Array(T, 1, 1); w[1] = W
    b = Array(T, 1, 1); b[1] = B
    u = Array(T, 1);    u[1] = U
	return LinearGaussianCISSM(f, v, g, w, b, u)
end

#####################################################
# CORE METHODS
#####################################################
process_matrix(m::AbstractLinearGaussian, state) = m.F
observation_matrix(m::AbstractLinearGaussian, state) = m.G
control_matrix(m::LinearGaussianCISSM, state) = m.B
control_input(m::LinearGaussianCISSM, state) = m.u

"""
Forecast the state of the process at the next time step.

#### Parameters
- m : AbstractGaussianSSM.  Model of the state evolution and observation processes.
- x : AbstractMvNormal.  Current state estimate.

#### Returns
- MvNormal distribution representing the forecast and its associated uncertainty.
"""
function predict(m::LinearGaussianCISSM, x::AbstractMvNormal)
    F = process_matrix(m, x)
    B = control_matrix(m, x)
    u = control_input(m, x)
    return MvNormal(F * mean(x) + B * u, F * cov(x) * F' + m.V)
end
