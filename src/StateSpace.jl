module StateSpace
using Distributions
using ForwardDiff
import Distributions: mean, var, cov, rand

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
include("filtered_states.jl")
include("filter_types.jl")
include("common.jl")

end # module
