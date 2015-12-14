abstract AbstractStateSpaceFilter

abstract AbstractKalmanFilter <: AbstractStateSpaceFilter
abstract LinearKalmanFilter <: AbstractKalmanFilter
abstract NonlinearKalmanFilter <: AbstractKalmanFilter
abstract NonlinearFilter <: AbstractStateSpaceFilter

# LinearGaussianSSM 	LinearKalmanFilter
# 						NonlinearKalmanFilter
# 						NonlinearFilter
#
# NonlinearGaussianSSM	NonlinearKalmanFilter
# 						NonlinearFilter
#
# NonlinearSSM 			NonlinearFilter

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


######################################################################
# Unscented Kalman filter
######################################################################


type UnscentedKalmanFilter{T<:Real} <: NonlinearKalmanFilter
	α::T
	β::T
	κ::T
end
typealias UKF UnscentedKalmanFilter
UnscentedKalmanFilter{T}(α::T=1e-3, β::T=2.0, κ::T=0.0) = UKF(α, β, κ)


"""
Data structure containing sigma points and their respective weights for the Unscented Kalman Filter

#### Fields
- `χ` : The matrix containing the 2L+1 sigma vectors, where L is the length of the system state vector.
- `wm` : Vector containing the weights for the corresponding sigma vectors required to reconsruct the predicted mean.
- `wc` : Vector containing the weights for the corresponding sigma vectors required to reconsruct the predicted covariance.
"""
type SigmaPoints{T<:Real}
    χ::Matrix{T}
    wm::Vector{T}
    wc::Vector{T}
end

"""
# calcSigmaPoints
Function to calculate the sigma points for the Unscented Transform

`calcSigmaPoints(state, α, β, κ)`
#### Parameters
- `state` : AbstractMvNormal type representing the current state estimate (mean vector with covariance matrix).
- `α` : Determines the spread of the sigma points around the mean of the system state. It is usually set to a small positive value. The default is set to 1e-3.
- `β` : Is used to incorporate prior knowledge of the distribution of the state. For Gaussian distributions a value of 2.0 is optimal. The default is set to 2.0.
- `κ` : Is a secondary scaling parameter which also determines the spread of sigma points around the mean of the system. This parameter allows for additional 'fine tuning'. It is usually set to a small value. The default is set to 0.0

`calcSigmaPoints(state, filter)`
#### Parameters
- `state` : AbstractMvNormal type representing the current state estimate (mean vector with covariance matrix).
- `filter` : UKFParameters type containing the α, β and κ parameters for the Unscented Kalman Filter.
"""
function calcSigmaPoints{T}(state::AbstractMvNormal, α::T, β::T, κ::T)
    x = mean(state)
    p = cov(state)
    L = length(x)
    χ = zeros(L, 2L+1)
    wm = zeros(2L+1)
    wc = zeros(2L+1)
    λ = α^2 * (L+κ) - L
    χ[:,1] = x
    wm[1] = λ/(L+λ)
    wc[1] = wm[1] + (1 - α^2 + β)
    wm[2:end] = 1/(2*(L+λ))
    wc[2:end] = wm[2:end]
    sp = sqrt(L+λ)*full(chol(p))
    for i = 2:L+1
        χ[:,i] = x + sp[:,i-1]
        χ[:,i+L] = x - sp[:,i-1]
    end
    return SigmaPoints(χ, wm, wc)
end
calcSigmaPoints(state::AbstractMvNormal, filter::UKF) = calcSigmaPoints(state, filter.α, filter.β, filter.κ)

"""
# timeUpdate
Function to calculate the predicted mean and the predicted covariance given a UKF model and the sigma points with their corresponding weights

`timeUpdate(m, sp)`
#### Parameters
- `m` : NonlinearGaussianSSM type containing the parameters of the Unscented Kalman Filter model.
- `sp` : SigmaPoints type containing the matrix of sigma vectors and their corresponding weights
"""
function timeUpdate(m::NonlinearGaussianSSM, sp::SigmaPoints)
    L, M  = size(sp.χ)
    χ_x = zeros(L, M)
    x_pred = zeros(L)
    p_pred = zeros(L, L)
    for i in 1:2L+1
        χ_x[:,i] = m.f(sp.χ[:,i])
        x_pred += sp.wm[i] * χ_x[:,i]
    end
    for i in 1:2L+1
        p_pred += sp.wc[i] * (χ_x[:,i] - x_pred)*(χ_x[:,i] - x_pred)'
    end
    p_pred += m.V
    return MvNormal(x_pred, p_pred), SigmaPoints(χ_x, sp.wm, sp.wc)
end

function predict(m::NonlinearGaussianSSM, x::AbstractMvNormal, filter::UKF)
    sigPoints = calcSigmaPoints(x, filter)
    pred_state, new_sigPoints = timeUpdate(m, sigPoints)
    return pred_state, new_sigPoints
end


function update(m::NonlinearGaussianSSM, x::AbstractMvNormal, sp::SigmaPoints, y)
    obsLength = length(y)
    L, M = size(sp.χ)
    y_trans = zeros(obsLength, M)
    y_pred = zeros(obsLength)
    x_pred = mean(x)
    for i in 1:2L+1
        y_trans[:,i] = m.g(sp.χ[:,i])
        y_pred += sp.wm[i] * y_trans[:,i]
    end

    P_xy = zeros(L, obsLength)
    P_yy = zeros(obsLength, obsLength)
    for i in 1:2L+1
        resy = (y_trans[:,i] - y_pred)
        P_xy += sp.wc[i] * (sp.χ[:,i] - x_pred) * resy'
        P_yy += sp.wc[i] * resy * resy'
    end
    P_yy += m.W
    kalmanGain = P_xy * inv(P_yy)
    new_x = x_pred + kalmanGain * (y - y_pred)
    new_cov = cov(x) - kalmanGain * P_yy * kalmanGain'

    return MvNormal(new_x, new_cov)
end

function filter{T}(m::NonlinearGaussianSSM, y::Array{T}, x0::AbstractMvNormal,
	filter::UKF=UKF())
    x_filtered = Array(AbstractMvNormal, size(y, 2))
    loglik = 0.0 #NEED TO SORT OUT LOGLIKELIHOOD FOR UKF
	x_pred, sigma_points = predict(m, x0, filtder)
	x_filtered[1] = update(m, x_pred, sigma_points, y[:, 1])
	for i in 2:size(y, 2)
		x_pred, sigma_points = predict(m, x_filtered[i-1], filter)
		# Check for missing values in observation
		if any(isnan(y[:, i]))
            x_filtered[i] = x_pred
        else
            x_filtered[i] = update(m, x_pred, sigma_points, y[:, i])
        end
	end
	return FilteredState(y, x_filtered, loglik)
end
