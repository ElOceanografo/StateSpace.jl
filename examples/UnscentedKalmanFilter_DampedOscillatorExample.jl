# An additive zero mean Gaussian noise Unscented Kalman Filter example,
#shamelessly taken from a paper by Rambabu Kandepu, Bjarne Foss and Lars Imsland
#"Applying the unscented Kalman filter for nonlinear state estimation"
#Link to the pdf version of the paper is here:
#http://folk.ntnu.no/bjarnean/pubs/journals/journal-47.pdf
#The example is the Van der Pol oscillator (Section 4.1 on page 6 of the paper)
#so see this paper for the full details of the problem. Here we only state
#enough to set up the Kalman Filter problem and solve it.

#Here we import the required modules
using StateSpace
using Distributions
using Gadfly
using DataFrames
using Colors

################################################################################
#Section: Set the Unscented Kalman Filter Parameters
#-------------------------------------------------------------------------------
#Here we're going to set the process model and the observation model functions
#for the oscillator along with the corresponding covarience matrices.
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
processCovariance = 1e-1*[0.1 0;0 1e-3]

#Set the observation noise covariance
observationCovariance = 3e-2*eye(2)

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
numObs = 100
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
initial_guess = MvNormal([0.0,0.0], 1.0*eye(2))
#End Section: Set guess of initial state
################################################################################


################################################################################
#Section: Execute the Additive noise Unscented Kalman Filter
#-------------------------------------------------------------------------------
filtered_state = filter(ukfStateModel, noisyObs, initial_guess)
#End Section: Execute the Additive noise Unscented Kalman Filter
################################################################################


################################################################################
#Section: Plot results
#-------------------------------------------------------------------------------
x_data = 1:numObs
x1_array = Vector{Float64}(numObs)
x1Var_array = Vector{Float64}(numObs)
x1_Guess = initial_guess.μ[1]
x1Var_Guess = 2*sqrt(initial_guess.Σ.mat[1,1])
for i in x_data
    current_state = filtered_state.state[i]
    x1_array[i] = current_state.μ[1]
    x1Var_array[i] = 2*sqrt(current_state.Σ.mat[1,1])
end
df_fs = DataFrame(
    x = x_data*Δt,
    y = x1_array,
    ymin = x1_array - x1Var_array,
    ymax = x1_array + x1Var_array,
    f = "Filtered values"
    )

n = 3
getColors = distinguishable_colors(n, Color[LCHab(70, 60, 240)],
                                   transform=c -> deuteranopic(c, 0.5),
                                   lchoices=Float64[65, 70, 75, 80],
                                   cchoices=Float64[0, 50, 60, 70],
                                   hchoices=linspace(0, 330, 24))
oscillatorx1_state_plot = plot(
    layer(x=x_data*Δt, y=noisyObs[1,:], Geom.point, Theme(default_color=getColors[2])),
    layer(x=x_data*Δt, y=trueState[1,:], Geom.line, Theme(default_color=getColors[3])),
    layer(df_fs, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon),
    Guide.xlabel("Time (seconds)"), Guide.ylabel("x1"),
    Guide.manual_color_key("Colour Key",["Filtered Estimate", "Measurements","True Value "],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Unscented Kalman Filter (Additive) Example")
    )
display(oscillatorx1_state_plot)
#End Section: Plot results
################################################################################


################################################################################
#Section: Perform Unscented Kalman Smoother
#-------------------------------------------------------------------------------
#Everything can be performed much better in hindsight and that's essentially
#where the Unscented Kalman smoother comes in. Given all of the observations and
#the filtered state (including the Kalman filter parameters, the smoother
#attempts to give better estimates of the system's state. Here we perform
#smoothing on the filtered data.
smoothed_state = smooth(ukfStateModel, filtered_state)
#End Section: Perform Unscented Kalman Smoother
################################################################################

################################################################################
#Section: Plot results
#-------------------------------------------------------------------------------
x_data = 1:numObs
x1_array = Vector{Float64}(numObs)
x1Var_array = Vector{Float64}(numObs)
x1_Guess = initial_guess.μ[1]
x1Var_Guess = 2*sqrt(initial_guess.Σ.mat[1,1])
for i in x_data
    current_state = smoothed_state.state[i]
    x1_array[i] = current_state.μ[1]
    x1Var_array[i] = 2*sqrt(current_state.Σ.mat[1,1])
end
df_fs = DataFrame(
    x = x_data*Δt,
    y = x1_array,
    ymin = x1_array - x1Var_array,
    ymax = x1_array + x1Var_array,
    f = "Smoothed values"
    )

n = 3
getColors = distinguishable_colors(n, Color[LCHab(70, 60, 240)],
                                   transform=c -> deuteranopic(c, 0.5),
                                   lchoices=Float64[65, 70, 75, 80],
                                   cchoices=Float64[0, 50, 60, 70],
                                   hchoices=linspace(0, 330, 24))
oscillatorx1_state_plot_smooted = plot(
    layer(x=x_data*Δt, y=noisyObs[1,:], Geom.point, Theme(default_color=getColors[2])),
    layer(x=x_data*Δt, y=trueState[1,:], Geom.line, Theme(default_color=getColors[3])),
    layer(df_fs, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon),
    Guide.xlabel("Time (seconds)"), Guide.ylabel("x1"),
    Guide.manual_color_key("Colour Key",["Filtered Estimate", "Measurements","True Value "],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Unscented Kalman Smoother (Additive) Example")
    )
display(oscillatorx1_state_plot_smooted)
#End Section: Plot results
################################################################################
