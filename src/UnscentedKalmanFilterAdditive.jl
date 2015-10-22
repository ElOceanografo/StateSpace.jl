"""
Data structure representing state space model

#### Fields
- `f` : The process model function relating the previous state of the system to the current state.
- `V` : N x N covariance matrix for the process
- `g` : The observation model function. This describes how the observation is generated from the system state at the current time
- `W` : M x M covariance matrix for the observation
"""
type AdditiveNonLinUKFSSM{T} <: AbstractSSM
    f::Function # Process function
	V::Matrix{T} # Process convariance
	g::Function # Observation function
	W::Matrix{T} # Observation covariance
end

function AdditiveNonLinUKFSSM{T}(f::Function, V::Matrix{T},
		g::Function, W::Matrix{T})
    return AdditiveNonLinUKFSSM{T}(f, V, g, W)
end

"""
Data structure containing the parameters required to place the sigma points for the Unscented Kalman Filter

#### Fields
- `α` : Determines the spread of the sigma points around the mean of the system state. It is usually set to a small positive value. The default is set to 1e-3.
- `β` : Is used to incorporate prior knowledge of the distribution of the state. For Gaussian distributions a value of 2.0 is optimal. The default is set to 2.0
- `κ` : Is a secondary scaling parameter which also determines the spread of sigma points around the mean of the system. This parameter allows for additional 'fine tuning'. It is usually set to a small value. The default is set to 0.0
"""
type UKFParameters{T<:Real}
    α::T
    β::T
    κ::T
end

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

`calcSigmaPoints(state, params)`
#### Parameters
- `state` : AbstractMvNormal type representing the current state estimate (mean vector with covariance matrix).
- `params` : UKFParameters type containing the α, β and κ parameters for the Unscented Kalman Filter.
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
calcSigmaPoints(state::AbstractMvNormal, params::UKFParameters) = calcSigmaPoints(state, params.α, params.β, params.κ)

"""
# timeUpdate
Function to calculate the predicted mean and the predicted covariance given a UKF model and the sigma points with their corresponding weights

`timeUpdate(m, sp)`
#### Parameters
- `m` : AdditiveNonLinUKFSSM type containing the parameters of the Unscented Kalman Filter model.
- `sp` : SigmaPoints type containing the matrix of sigma vectors and their corresponding weights
"""
function timeUpdate(m::AdditiveNonLinUKFSSM, sp::SigmaPoints)
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

function predict(m::AdditiveNonLinUKFSSM, x::AbstractMvNormal, params::UKFParameters)
    sigPoints = calcSigmaPoints(x, params)
    pred_state, new_sigPoints = timeUpdate(m, sigPoints)
    return pred_state, new_sigPoints
end

function observe(m::AdditiveNonLinUKFSSM, x::AbstractMvNormal, sp::SigmaPoints, y)
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
    P_yy += m.W
    return MvNormal(y_pred, P_yy), P_xy
end

function observe(m::AdditiveNonLinUKFSSM, x::AbstractMvNormal, sp::SigmaPoints, y, meas_cov::Matrix)
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
    P_yy += meas_cov
    return MvNormal(y_pred, P_yy), P_xy
end

function innovate(m::AdditiveNonLinUKFSSM, x::AbstractMvNormal, yPred::AbstractMvNormal, P_xy::Matrix, sp::SigmaPoints, y::Vector)
    kalmanGain = P_xy * inv(cov(yPred))
    new_x = mean(x) + kalmanGain * (y - mean(yPred))
    new_cov = cov(x) - kalmanGain * cov(yPred) * kalmanGain'
    return MvNormal(new_x, new_cov)
end

function update(m::AdditiveNonLinUKFSSM, x::AbstractMvNormal, sp::SigmaPoints, y::Vector)
    yPred, P_xy = observe(m, x, sp, y)
    return innovate(m, x, yPred, P_xy, sp, y)
end

function update(m::AdditiveNonLinUKFSSM, x::AbstractMvNormal, sp::SigmaPoints, y::Vector, meas_cov::Matrix)
    yPred, P_xy = observe(m, x, sp, y, meas_cov)
    return innovate(m, x, yPred, P_xy, sp, y)
end

function filter{T}(m::AdditiveNonLinUKFSSM, y::Array{T}, x0::AbstractMvNormal, estMissObs::Bool=false, α::T=1e-3, β::T=2.0, κ::T=0.0)
    params = UKFParameters(α, β, κ)
    x_filtered = Array(AbstractMvNormal, size(y, 2) + 1)
	x_filtered[1] = x0
    y_obs = zeros(y)
    loglik = 0.0
# 	x_pred, sigma_points = predict(m, x0, params)
# 	x_filtered[1] = update(m, x_pred, sigma_points, y[:, 1])
	for i in 1:size(y, 2)
        y_current = y[:, i]
		x_pred, sigma_points = predict(m, x_filtered[i], params)
        y_pred, P_xy = observe(m, x_pred, sigma_points, y_current)
		# Check for missing values in observation
        y_Boolean = isnan(y_current)
        if any(y_Boolean)
            if estMissObs
                y_current, y_cov_mat = estimateMissingObs(m, x_pred, y_pred, y_current, y_Boolean)
                x_filtered[i+1] = update(m, x_pred, sigma_points, y_current, y_cov_mat)
                loglik += logpdf(observe(m, x_filtered[i+1], calcSigmaPoints(x_filtered[i+1], params), y_current)[1], y_current)
            else
                x_filtered[i+1] = x_pred
            end
        else
            x_filtered[i+1] = update(m, x_pred, sigma_points, y_current)
			loglik += logpdf(observe(m, x_filtered[i+1], calcSigmaPoints(x_filtered[i+1], params), y_current)[1], y_current)
        end
        loglik += logpdf(x_pred, mean(x_filtered[i+1]))
        y_obs[:,i] = y_current
	end
	return FilteredState(y_obs, x_filtered, loglik)
end

function smoothedTimeUpdate(m::AdditiveNonLinUKFSSM, currentState::AbstractMvNormal, sp::SigmaPoints)
    L, M  = size(sp.χ)
    χ_x = zeros(L, M)
    x_pred = zeros(L)
    p_pred = zeros(L, L)
    cross_cov = zeros(L, L)
    for i in 1:2L+1
        χ_x[:,i] = m.f(sp.χ[:,i])
        x_pred += sp.wm[i] * χ_x[:,i]
    end
    for i in 1:2L+1
        p_pred += sp.wc[i] * (χ_x[:,i] - x_pred)*(χ_x[:,i] - x_pred)'
        cross_cov += sp.wc[i] * (sp.χ[:,i] - mean(currentState)) * (χ_x[:,i] - x_pred)'
    end
    p_pred += m.V
    return MvNormal(x_pred, p_pred), cross_cov
end

function predictSmooth(m::AdditiveNonLinUKFSSM, x::AbstractMvNormal, params::UKFParameters)
    sigPoints = calcSigmaPoints(x, params)
    return smoothedTimeUpdate(m, x, sigPoints)[1]
end

function smooth{T}(m::AdditiveNonLinUKFSSM, fs::FilteredState, α::T=1e-3, β::T=2.0, κ::T=0.0)
    params = UKFParameters(α, β, κ)
	n = size(fs.observations, 2)
	smooth_dist = Array(AbstractMvNormal, n)
	smooth_dist[end] = fs.state[end]
    loglik = logpdf(observe(m, smooth_dist[end], calcSigmaPoints(smooth_dist[end], params), fs.observations[:, end])[1], fs.observations[:, end])
	for i in (n - 1):-1:1
		sp = calcSigmaPoints(fs.state[i], params)
        pred_state, cross_covariance = smoothedTimeUpdate(m, fs.state[i], sp)
        smootherGain = cross_covariance * inv(cov(pred_state))
        x_smooth = mean(fs.state[i]) + smootherGain * (mean(smooth_dist[i+1]) - mean(pred_state))
        P_smooth = cov(fs.state[i]) + smootherGain * (cov(smooth_dist[i+1]) - cov(pred_state)) * smootherGain'
		smooth_dist[i] = MvNormal(x_smooth, P_smooth)
        loglik += logpdf(predictSmooth(m, smooth_dist[i], params), mean(smooth_dist[i+1]))
		if !any(isnan(fs.observations[:, i]))
            loglik += logpdf(observe(m, smooth_dist[i], calcSigmaPoints(smooth_dist[i], params), fs.observations[:, i])[1], fs.observations[:, i])
		end
	end
	return SmoothedState(fs.observations, smooth_dist, loglik)
end
