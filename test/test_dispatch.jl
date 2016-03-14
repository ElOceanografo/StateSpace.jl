println("Testing method dispatch...")

# LinearGaussianSSM
#	- LinearKalmanFilter
@assert method_exists(predict, (LinearGaussianSSM, AbstractMvNormal))
@assert method_exists(update, (LinearGaussianSSM, AbstractMvNormal, Vector))
@assert method_exists(update, 
	(LinearGaussianSSM, AbstractMvNormal, Vector, Int))
@assert method_exists(update, 
	(LinearGaussianSSM, AbstractMvNormal, Vector, KalmanFilter, Int))

# NonlinearGaussianSSM
#	- NonlinearKalmanFilter
@assert method_exists(update,
	(NonlinearGaussianSSM, AbstractMvNormal, Vector))

for T in [NonlinearKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter]
	println(T)
	@assert method_exists(update, 
		(NonlinearGaussianSSM, AbstractMvNormal, Vector, T))
	@assert method_exists(update, 
		(NonlinearGaussianSSM, AbstractMvNormal, Vector, T, Int))
end

# Special predict methods that don't return just a Distribution
@assert method_exists(predict, (NonlinearGaussianSSM, AbstractMvNormal, UKF))
@assert method_exists(predict, (NonlinearGaussianSSM, Matrix, EnKF))

@assert method_exists(update, 
	(NonlinearGaussianSSM, Matrix, Vector, EnKF))

# NonlinearSSM 			NonlinearFilter


println("Dispatch passed.\n")