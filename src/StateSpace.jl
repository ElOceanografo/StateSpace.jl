module StateSpace

import Base: mean, filter, show
import StatsBase: loglikelihood


export
	AbstractStateSpaceModel,
	AbstractSSM,
	AbstractGaussianSSM,
	LinearGaussianSSM,
	LinearGaussianCISSM,
	NonlinearGaussianSSM,
	FilteredState,
	show,
	process_matrix,
	observation_matrix,
	predict,
	observe,
	update,
	update!,
	filter,
	smooth,
	loglikelihood,
	simulate

include("common.jl")
include("KalmanFilter.jl")
include("ExtendedKalmanFilter.jl")

end # module
