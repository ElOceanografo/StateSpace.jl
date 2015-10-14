module StateSpace

import Base: mean, filter, show
import StatsBase: loglikelihood


export
	AbstractStateSpaceModel,
	AbstractSSM,
	AbstractGaussianSSM,
    AdditiveNonLinUKFSSM,
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
    innovate,
	filter,
	smooth,
    estimateMissingObs!,
	loglikelihood,
	simulate

include("common.jl")
include("KalmanFilter.jl")
include("ExtendedKalmanFilter.jl")
include("UnscentedKalmanFilterAdditive.jl")

end # module
