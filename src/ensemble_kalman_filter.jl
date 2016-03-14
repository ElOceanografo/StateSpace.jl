
type EnsembleKalmanFilter{I<:Integer} <: NonlinearFilter
	nparticles::I
end
typealias EnKF EnsembleKalmanFilter


function predict(m::AbstractGaussianSSM, ensemble::Matrix, filter::EnKF=EnKF(); 
		u::Vector=zeros(m.nu), t::Int=1)
	ensemble_new = similar(ensemble)
    CI = control_input(m, u, t)
	for i in 1:size(ensemble, 2)
		F = process_matrix(m, x[:, i], t)
		ensemble_new[:, i] = F * ensemble[:, i] + CI
	end
	return ensemble_new
end

function update(m::AbstractGaussianSSM, ensemble::Matrix, y::Vector,
		filter::EnKF=EnKF())
	x = mean(ensemble, 2)
	V = cov(ensemble')
	return update_kalman(m, MvNormal(x, V), y)
end

