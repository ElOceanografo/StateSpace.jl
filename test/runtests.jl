using StateSpace
using Distributions
using Base.Test


function random_cov(n)
	A = rand(n, n)
	A = A + A'
	return A + n * eye(n)
end

include("test_dispatch.jl")
include("test_KF.jl")
include("test_EKF.jl")
include("test_UKF.jl")
include("test_EnKF.jl")
include("test_particle_filter.jl")
println("Passed all tests.")