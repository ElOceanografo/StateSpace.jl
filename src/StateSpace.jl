module StateSpace

import Base: mean, filter, show
import StatsBase: loglikelihood


export
	LinearGaussianSSM,
	NonlinearGaussianSSM,
	FilteredState,
	show,
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
