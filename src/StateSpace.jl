module StateSpace

import Base: mean, filter, show
import StatsBase: loglikelihood


export
	AbstractStateSpaceModel,
	AbstractSSM,
	AbstractGaussianSSM,
    AdditiveNonLinUKFSSM,
	LinearGaussianSSM,
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

include("matrix_utils.jl")
include("model_types.jl")
include("common.jl")
include("kalman_filter.jl")
include("ExtendedKalmanFilter.jl")
include("UnscentedKalmanFilterAdditive.jl")

end # module
