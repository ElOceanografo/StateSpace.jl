println("Testing method dispatch...")

# LinearGaussianSSM
#	- LinearKalmanFilter
@assert method_exists(update, (LinearGaussianSSM, AbstractMvNormal, Vector))
@assert method_exists(update, 
	(LinearGaussianSSM, AbstractMvNormal, Vector, Int))
@assert method_exists(update, 
	(LinearGaussianSSM, AbstractMvNormal, Vector, KalmanFilter, Int))

# NonlinearGaussianSSM
#	- NonlinearKalmanFilter

for T in [NonlinearKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter]
	@assert method_exists(update,
		(NonlinearGaussianSSM, AbstractMvNormal, Vector))
	@assert method_exists(update, 
		(NonlinearGaussianSSM, AbstractMvNormal, Vector, T))
	@assert method_exists(update, 
		(NonlinearGaussianSSM, AbstractMvNormal, Vector, T, Int))
end

# NonlinearSSM 			NonlinearFilter


println("Dispatch passed.\n")