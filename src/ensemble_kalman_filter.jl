
type EnsembleKalmanFilter{T} <: NonlinearFilter end
typealias EnKF EnsembleKalmanFilter

function predict(m::AbstractGaussianSSM, ensemble::Matrix, filter::EnKF=EnKF())
	ensemble_new = zeros(size(ensemble))
	for i in 1:size(ensemble, 2)
		ensemble_new[:, i] = m.f(ensemble[:, i])
	end
	return ensemble_new
end

function update(m::AbstractGaussianSSM, ensemble::Matrix, y::Vector,
		filter::EnKF=EnKF())
	x = mean(ensemble, 2)
	V = cov(ensemble')
	return update_kalman(m, MvNormal(x, V), y)
end
