abstract AbstractStateSpaceFilter

abstract AbstractKalmanFilter <: AbstractStateSpaceFilter
abstract LinearKalmanFilter <: AbstractKalmanFilter
abstract NonlinearKalmanFilter <: AbstractKalmanFilter
abstract NonlinearFilter <: AbstractStateSpaceFilter

# LinearGaussianSSM 	LinearKalmanFilter
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

# basic Kalman update, once we have the predicted state, an observation,
# and the observation matrix/covariance.
function update_kalman(pred, y, G, W)
    innovation = y - G * mean(pred)
    innovation_cov = G * cov(pred) * G' + W
    K = cov(pred) * G' * inv(innovation_cov)
    mean_update = mean(pred) + K * innovation
    cov_update = (eye(cov(pred)) - K * G) * cov(pred)
    return MvNormal(mean_update, cov_update)
end

function update(m::LinearGaussianSSM, pred::AbstractMvNormal, y::Vector,
		filter::KalmanFilter=KalmanFilter(), t::Int=1)
	return update_kalman(pred, y, m.G(t), m.W(t))
end

function update(m::LinearGaussianSSM, pred::AbstractMvNormal, y::Vector, t::Int=1)
    return update(m, pred, y, KalmanFilter(), t)
end

function update!(m::LinearGaussianSSM, fs::FilteredState, y::Vector;
		u::Vector=zeros(m.nu), filter::KalmanFilter=KalmanFilter(), t::Int=1)
	x_pred = predict(m, fs.state[end], u=u, t=t-1)
	x_filt = update_kalman(x_pred, y, m.G(t), m.W(t))
	push!(fs.state, x_filt)
	fs.observations = [fs.observations y]
end


######################################################################
# Extended Kalman filter
######################################################################
type ExtendedKalmanFilter <: NonlinearKalmanFilter end
typealias EKF ExtendedKalmanFilter


## methods

function update(m::NonlinearGaussianSSM, pred::AbstractMvNormal, y::Vector,
		filter::NonlinearKalmanFilter=EKF(), t::Int=1)
    G = observation_matrix(m, pred, t)
    W = m.W(t)
	return update_kalman(pred, y, G, W)
end

function update!(m::NonlinearGaussianSSM, fs::FilteredState, y::Vector;
		u::Vector=zeros(m.nu), filter::NonlinearKalmanFilter=EKF(), t::Int=1)
	x_pred = predict(m, fs.state[end], u=u, t=t-1)
    G = observation_matrix(m, x_pred, t)
    W = m.W(t)
	x_filt = update_kalman(x_pred, y, G, W)
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
- `m` : AbstractGaussianSSM type containing the parameters of the Unscented Kalman Filter model.
- `sp` : SigmaPoints type containing the matrix of sigma vectors and their corresponding weights
"""
function timeUpdate(m::AbstractGaussianSSM, sp::SigmaPoints, t::Int=1)
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
    p_pred += m.V(t)
    return MvNormal(x_pred, p_pred), SigmaPoints(χ_x, sp.wm, sp.wc)
end

function predict(m::AbstractGaussianSSM, x::AbstractMvNormal, filter::UKF)
    sigPoints = calcSigmaPoints(x, filter)
    pred_state, new_sigPoints = timeUpdate(m, sigPoints)
    return pred_state, new_sigPoints
end

function observe(m::AbstractGaussianSSM, x::AbstractMvNormal, sp::SigmaPoints, y, t::Int=1)
    obsLength = length(y)
    L, M = size(sp.χ)
    y_trans = zeros(obsLength, M)
    y_pred = zeros(obsLength)
    for i in 1:2L+1
        y_trans[:,i] = m.g(sp.χ[:,i])
        y_pred += sp.wm[i] * y_trans[:,i]
    end

    P_xy = zeros(L, obsLength)
    P_yy = zeros(obsLength, obsLength)
    for i in 1:2L+1
        resy = (y_trans[:,i] - y_pred)
        P_xy += sp.wc[i] * (sp.χ[:,i] - mean(x)) * resy'
        P_yy += sp.wc[i] * resy * resy'
    end
    P_yy += m.W(t)
    return MvNormal(y_pred, P_yy), P_xy
end

function innovate(m::AbstractGaussianSSM, x::AbstractMvNormal, yPred::AbstractMvNormal, P_xy::Matrix, sp::SigmaPoints, y::Vector)
    kalmanGain = P_xy * inv(cov(yPred))
    new_x = mean(x) + kalmanGain * (y - mean(yPred))
    new_cov = cov(x) - kalmanGain * cov(yPred) * kalmanGain'
    return MvNormal(new_x, new_cov)
end

function update(m::AbstractGaussianSSM, x::AbstractMvNormal, sp::SigmaPoints, y::Vector)
    yPred, P_xy = observe(m, x, sp, y)
    return innovate(m, x, yPred, P_xy, sp, y)
end

function filter{T}(m::AbstractGaussianSSM, y::Array{T}, x0::AbstractMvNormal,
	filter::UKF=UKF())
    x_filtered = Array(AbstractMvNormal, size(y, 2))
    loglik = 0.0
	x_pred, sigma_points = predict(m, x0, filter)
	x_filtered[1] = update(m, x_pred, sigma_points, y[:, 1])
	for i in 2:size(y, 2)
		x_pred, sigma_points = predict(m, x_filtered[i-1], filter)
		# Check for missing values in observation
		if any(isnan(y[:, i]))
            x_filtered[i] = x_pred
        else
            x_filtered[i] = update(m, x_pred, sigma_points, y[:, i])
            loglik += logpdf(observe(m, x_filtered[i], calcSigmaPoints(x_filtered[i], filter.α, filter.β, filter.κ), y[:, 1])[1], y[:, 1])
        end
        loglik += logpdf(x_pred, mean(x_filtered[i]))
	end
	return FilteredState(y, x_filtered, loglik, false)
end

######################################################################
# Ensemble Kalman filter
######################################################################


type EnsembleKalmanFilter{I<:Integer} <: NonlinearFilter
    nparticles::I
end
typealias EnKF EnsembleKalmanFilter
EnsembleKalmanFilter() = EnsembleKalmanFilter(100)


function predict(m::AbstractGaussianSSM, ensemble::Matrix, filter::EnKF=EnKF();
        u::Vector=zeros(m.nu), t::Int=1)
    ensemble_new = similar(ensemble)
    CI = control_input(m, u, t)
    for i in 1:size(ensemble, 2)
        F = process_matrix(m, ensemble[:, i], t)
        ensemble_new[:, i] = F * ensemble[:, i] + CI
    end
    return ensemble_new
end

function update(m::AbstractGaussianSSM, ensemble::Matrix, y::Vector,
        filt::EnKF=EnKF(), t::Int=1)
    P = cov(ensemble')
    ensemble_updated = similar(ensemble)
    for i in 1:filt.nparticles
        G = observation_matrix(m, ensemble[:, i], t)
        W = observation_matrix(m, ensemble[:, i], t)
        innovation = y - G * ensemble[:, i]
        innovation_cov = G * P * G' + W
        K = P * G' * inv(innovation_cov)
        ensemble_updated[:, i] = ensemble[:, i] + K * innovation
        cov_update = (eye(P) - K * G) * P
    end
    return ensemble_updated
end

function filter{T}(m::AbstractGaussianSSM, y::Array{T}, x0::AbstractMvNormal,
        filt::EnKF=EnKF(); u::Matrix{T}=zeros(m.nu, size(y, 2)), return_ensemble=false)
    nt = size(y, 2)
    ensemble = rand(x0, filt.nparticles)
    ensemble = predict(m, ensemble, filt, u=u[:,1], t=1)
    loglik = 0.0
    if return_ensemble
        x_filtered = zeros(m.nx, filt.nparticles, nt)
        x_filtered[:, :, 1] = update(m, ensemble, y[:, 1], filt, 1)
        for i in 2:nt
            ensemble = predict(m, x_filtered[:, :, i-1], filt, u=u[:, i], t=i)
            x_filtered[:, :, i] = update(m, ensemble, y[:, i], filt, i)
        end
        return x_filtered
    else
        x_filtered = Array(AbstractMvNormal, size(y, 2))
        ensemble = update(m, ensemble, y[:, 1], filt, 1)
        x_filtered[1] = MvNormal(vec(mean(ensemble, 2)), cov(ensemble'))
        for j in 1:filt.nparticles
            loglik += logpdf(x_filtered[1], ensemble[:, j])
            loglik += logpdf(observe(m, x_filtered[1]), y[:,1])
        end
        for i in 2:nt
            ensemble = predict(m, ensemble, filt, u=u[:, i], t=i)
            x_filtered[i] = MvNormal(vec(mean(ensemble, 2)), cov(ensemble'))
            for j in 1:filt.nparticles
                loglik += logpdf(x_filtered[i], ensemble[:, j])
                loglik += logpdf(observe(m, x_filtered[i]), y[:,1])
            end
        end
        return FilteredState(y, x_filtered, loglik, false)
    end
end
