abstract AbstractStateSpaceFilter

abstract LinearKalmanFilter <: AbstractStateSpaceFilter
abstract NonlinearKalmanFilter <: AbstractStateSpaceFilter

######################################################################
# Kalman filter
######################################################################

type KalmanFilter <: LinearKalmanFilter end
typealias KF KalmanFilter

## methods
function update_kalman(m::AbstractGaussianSSM, pred::AbstractMvNormal,
	y::Vector, t::Int)
	G = observation_matrix(m, pred, t)
	innovation = y - G * mean(pred)
	innovation_cov = G * cov(pred) * G' + m.W(t)
	K = cov(pred) * G' * inv(innovation_cov)
	mean_update = mean(pred) + K * innovation
	cov_update = (eye(cov(pred)) - K * G) * cov(pred)
	return MvNormal(mean_update, cov_update)
end

function update(m::LinearGaussianSSM, pred::AbstractMvNormal, y::Vector;
		t::Int=1, filter::KalmanFilter=KalmanFilter())
	return update_kalman(m, pred, y, t)
end

function update!(m::LinearGaussianSSM, fs::FilteredState, y::Vector;
		u::Vector=zeros(m.nu), t::Int=1, filter::KalmanFilter=KalmanFilter())
	x_pred = predict(m, fs.state[end], u=u, t=t-1)
	x_filt = update_kalman(m, x_pred, y, t)
	push!(fs.state, x_filt)
	fs.observations = [fs.observations y]
end


######################################################################
# Extended Kalman filter
######################################################################
type ExtendedKalmanFilter <: NonlinearKalmanFilter end
typealias EKF ExtendedKalmanFilter


## methods

function update(m::NonlinearGaussianSSM, pred::AbstractMvNormal, y::Vector;
		t::Int=1, filter::NonlinearKalmanFilter=EKF())
	return update_kalman(m, pred, y, t)
end

function update!(m::NonlinearGaussianSSM, fs::FilteredState, y::Vector;
		u::Vector=zeros(m.nu), t::Int=1, filter::NonlinearKalmanFilter=EKF())
	x_pred = predict(m, fs.state[end], u=u, t=t-1)
	x_filt = update_kalman(m, x_pred, y, t)
	push!(fs.state, x_filt)
	fs.observations = [fs.observations y]
end
