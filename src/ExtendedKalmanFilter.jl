using ForwardDiff

type NonlinearGaussianSSM{T} <: AbstractGaussianSSM
	f::Function # actual process function
	fjac::Function # function returning Jacobian of process
	V::Matrix{T}
	g::Function # actual observation function
	gjac::Function # function returning Jacobian of process
	W::Matrix{T}
end

function NonlinearGaussianSSM{T}(f::Function, V::Matrix{T}, 
		g::Function, W::Matrix{T})
	fjac = jacobian(f)
	gjac = jacobian(g)
	return NonlinearGaussianSSM{T}(f, fjac, V, g, gjac, W)
end


## Core methods
process_matrix(m::NonlinearGaussianSSM, x::AbstractMvNormal) = m.fjac(mean(x))
process_matrix(m::NonlinearGaussianSSM, x::Vector) = m.fjac(x)
observation_matrix(m::NonlinearGaussianSSM, x::AbstractMvNormal) = m.gjac(mean(x))
observation_matrix(m::NonlinearGaussianSSM, x::Vector) = m.gjac(x)
