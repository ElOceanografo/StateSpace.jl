type AdditiveNonLinUKFSSM{T} <: AbstractSSM
    f::Function # Process function
	V::Matrix{T} # Process convariance
	g::Function # Observation function
	W::Matrix{T} # Observation covariance
end

type UKFParameters{T<:Real}
    α::T
    β::T
    κ::T
end

type SigmaPoints{T<:Real}
    χ::Matrix{T}
    wm::Vector{T}
    wc::Vector{T}
end

function AdditiveNonLinUKFSSM{T}(f::Function, V::Matrix{T},
		g::Function, W::Matrix{T})
    return AdditiveNonLinUKFSSM{T}(f, V, g, W)
end


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

function update(m::AdditiveNonLinUKFSSM, x::AbstractMvNormal, sp::SigmaPoints, y)
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

function filter{T}(m::AdditiveNonLinUKFSSM, y::Array{T}, x0::AbstractMvNormal, α::T=1e-3, β::T=2.0, κ::T=0.0)
    params = UKFParameters(α, β, κ)
    x_filtered = Array(AbstractMvNormal, size(y, 2))
    loglik = 0.0 #NEED TO SORT OUT LOGLIKELIHOOD FOR UKF
	x_pred, sigma_points = predict(m, x0, params)
	x_filtered[1] = update(m, x_pred, sigma_points, y[:, 1])
	for i in 2:size(y, 2)
		x_pred, sigma_points = predict(m, x_filtered[i-1], params)
		# Check for missing values in observation
		if any(isnan(y[:, i]))
            x_filtered[i] = x_pred
        else
            x_filtered[i] = update(m, x_pred, sigma_points, y[:, i])
        end
	end
	return FilteredState(y, x_filtered, loglik)
end