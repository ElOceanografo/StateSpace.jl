println("Testing UnscentedKalmanFilterAdditive.jl...")

Δt = 0.1 # Set the time step

#Set the process function
function processFunction(x::Vector, Δt::Float64=0.1, μ::Float64=0.3)
    x1 = zeros(x)
    x1[1] = x[1] + Δt * -x[2]
    x1[2] = x[2] + Δt * (-μ * (1 - x[1]^2) * x[2] + x[1])
    x1
end

#Set the observation function.
observationFunction(x::Vector) = x

#Set process noise covariance matrix
processCovariance = 1e-2*[0.1 0;
    0 1e-3]

#Set the observation noise covariance
observationCovariance = 1e-1*eye(2)

#Create additive noise UKF model
ukfStateModel = AdditiveNonLinUKFSSM(processFunction, processCovariance,
observationFunction, observationCovariance)
#End Section: Set the Unscented Kalman Filter Parameters
################################################################################


################################################################################
#Section: Generate noisy observations
#-------------------------------------------------------------------------------
#Set the true initial state of the oscillator
trueInitialState = [1.4, 0.0]

#Set the number of observations and generate the true values and the noisy
#observations.
numObs = 200
trueState = zeros(2,numObs)
noisyObs = zeros(2,numObs)
trueState[:,1] = trueInitialState
noisyObs[:,1] = trueInitialState + sqrt(observationCovariance)*randn(2)
for i in 2:numObs
    trueState[:,i] = processFunction(trueState[:,i-1])
    noisyObs[:,i] = trueState[:,i] + sqrt(observationCovariance)*randn(2)
end
#End Section: Generate noisy observations
################################################################################

################################################################################
#Section: Set guess of initial state
#-------------------------------------------------------------------------------
initial_guess = MvNormal([0.0,5.0], 5.0*eye(2))
#End Section: Set guess of initial state
################################################################################


################################################################################
#Section: Execute the Additive noise Unscented Kalman Filter
#-------------------------------------------------------------------------------
filtered_state = filter(ukfStateModel, noisyObs, initial_guess)
#End Section: Execute the Additive noise Unscented Kalman Filter
################################################################################


println("UnscentedKalmanFilterAdditive.jl passed.\n")
