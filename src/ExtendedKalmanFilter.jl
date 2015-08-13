using ForwardDiff

type NonlinearGaussianSSM{T} <: AbstractGaussianSSM
	f::Function # actual process function
	m::Int64
	fjac::Function # function returning Jacobian of process
	V::Matrix{T}
	g::Function # actual observation function
	n::Int64
	gjac::Function # function returning Jacobian of process
	W::Matrix{T}
end

function NonlinearGaussianSSM{T}(f::Function, m::Int64, V::Matrix{T}, 
		g::Function, n::Int64, W::Matrix{T})
	fjac = forwarddiff_jacobian(f, T, fadtype=:typed)
	gjac = forwarddiff_jacobian(g, T, fadtype=:typed)
	return NonlinearGaussianSSM{T}(f, m, fjac, V, g, n, gjac, W)
end


## Core methods
process_matrix(m::NonlinearGaussianSSM, x::AbstractMvNormal) = m.fjac(mean(x))
process_matrix(m::NonlinearGaussianSSM, x::Vector) = m.fjac(x)
observation_matrix(m::NonlinearGaussianSSM, x::AbstractMvNormal) = m.gjac(mean(x))
observation_matrix(m::NonlinearGaussianSSM, x::Vector) = m.gjac(x)
