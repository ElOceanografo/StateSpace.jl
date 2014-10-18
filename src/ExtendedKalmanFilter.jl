using ForwardDiff

type NonlinearGaussianSSM{T}
	f::Function # actual process function
	m::Int64
	fjac::Function # function returning Jacobian of process
	V::Matrix{T}
	g::Function # actual observation function
	n::Int64
	gjac::Function # function returning Jacobian of process
	W::Matrix{T}
end

function NonlinearGaussianSSM{T}(f::Function, m::Int64, V::Matrix{T}, 
		g::Function, n::Int64, W::Matrix{T})
	fjac = forwarddiff_jacobian(f, T, fadtype=:typed)
	gjac = forwarddiff_jacobian(g, T, fadtype=:typed)
	return NonlinearGaussianSSM{T}(f, m, fjac, V, g, n, gjac, W)
end


## Core methods
function predict(m::NonlinearGaussianSSM, x::GenericMvNormal)
	F = m.fjac(mean(x))
	return MvNormal(m.f(mean(x)), F * cov(x) * F' + m.V)
end

function observation(m::NonlinearGaussianSSM, x::GenericMvNormal)
	return MvNormal(m.g(mean(x)), m.W)
end

function update(m::NonlinearGaussianSSM, pred::GenericMvNormal, y)
	G = m.gjac(mean(pred))
	innovation = y - G * mean(pred)
	innovation_cov = G * cov(pred) * G' + m.W
	K = cov(pred) * G' * inv(innovation_cov)
	mean_update = mean(pred) + K * innovation
	cov_update = (eye(cov(pred)) - K * G) * cov(pred)
	return MvNormal(mean_update, cov_update)
end

function update!(m::NonlinearGaussianSSM, fs::FilteredState, y)
	x_pred = predict(m, fs.state_dist[end])
	x_filt = update(m, x_pred, y)
	push!(fs.state_dist, x_filt)
	fs.observations = [fs.observations y]
	return fs
end


function filter{T}(y::Array{T}, m::NonlinearGaussianSSM{T}, x0::GenericMvNormal)
	x_filtered = Array(GenericMvNormal, size(y, 2))
	loglik = 0
	x_pred = predict(m, x0)
	x_filtered[1] = update(m, x_pred, y[:, 1])
	for i in 2:size(y, 2)
		x_pred = predict(m, x_filtered[i-1])
		# Check for missing values in observation
		if any(isnan(y[:, i]))
			x_filtered[i] = x_pred
		else
			x_filtered[i] = update(m, x_pred, y[:, i])
			loglik += log(pdf(observation(m, x_filtered[i]), y[:, i]))
		end
		# this may not be right...double check definition
		loglik += log(pdf(x_pred, mean(x_filtered[i])))
	end
	return FilteredState(y, x_filtered, loglik)
end

function smooth{T}(fs::FilteredState{T})
	error("Not implemented yet")
end

function simulate(m::NonlinearGaussianSSM, n::Int64, x0::GenericMvNormal)
	x = zeros(length(x0), n)
	y = zeros(size(m.G, 1), n)
	x[:, 1] = rand(MvNormal(m.f(mean(x0)), m.V))
	y[:, 1] = rand(MvNormal(m.g(x[:, 1]), m.W))
	for i in 2:n
		x[:, i] = rand(MvNormal(m.f(x[:, i-1]), m.V))
		y[:, i] = rand(MvNormal(m.g(x[:, i]), m.W))
	end
	return (x, y)
end
