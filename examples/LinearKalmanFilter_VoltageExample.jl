#Linear Kalman filter example
#This example closely follows that given on "Greg Czerniak's Website". Namely
#the voltage example on this page: http://greg.czerniak.info/guides/kalman1/

#Let's import the modules required to execute the Kalman Filter and visualize
#the results
using StateSpace
using Distributions
using DataFrames
using Gadfly
using Colors

#To use the Kalman Filter we need three things:
#1) Some observations
#2) The parameters of the Kalman Filter model describing the system process and
#the observation model.
#3) An initial guess of the state.
#
#The following code is split into those three sections to begin with. We finish
#by performing the Kalman Filter algorithm and finally displaying the results.

################################################################################
#Section: Generate noisy Observations
#-------------------------------------------------------------------------------
#Let's assume that the true voltage value does not change throughout the
#experiment. So we can give the true voltage a value
true_voltage = 1.25

#However our measurements of the true voltage are noisy. Here we assume that the
#noise is normally distributed with zero mean but non-zero variance:
measurement_noise_variance = 0.1

#Given this information we can generate some noisy data (observations). We'll
#also choose the number of observations
number_of_observations = 60
observations = randn(number_of_observations) * sqrt(measurement_noise_variance) + true_voltage

#Note that the observation list has to be transposed because each column of the
#matrix is considered as a new observation.
observations = observations'
#End Section: Generate noisy Observations
################################################################################

################################################################################
#Section: Describe Kalman Filter parameters
#-------------------------------------------------------------------------------
#Now that we have some observations we need to describe the Kalman Filter
#parameters:
#Process matrix - Note that since this is a one-dimensional system I've given
#a floating point value rather than explicitly declaring a matrix with one
#element, although you can do it that way. The value of 1.0 here means that the
#process model suggests that the previous state is the same as the next state.
#i.e. the voltage doesn't change.
process_matrix = 1.0

#Process Covariance - Since it's a single value, the covariance is the same as
#variance. We've made it small here because we are fairly sure of our process
#model that the state doesn't change. If we were less sure that the model was
#correct then we would increase this value.
process_covariance = 0.00001

#Observation matrix - Since we measure the voltage directly (we don't measure
#some quantity related by a function - i.e. we don't measure twice the voltage)
#the observation matrix is equal to 1.0
observation_matrix = 1.0

#Observation Covariance - we set the variance of the measurement noise earlier
#so we'll give it the same value.
observation_covariance = measurement_noise_variance

#Now we can create our Linear State Space Model
linSSM = LinearGaussianSSM(process_matrix, process_covariance, observation_matrix, observation_covariance)
#End Section: Describe Kalman Filter parameters
################################################################################

################################################################################
#Section: Set initial guess of the state
#-------------------------------------------------------------------------------
#We'll begin with a guess that's quite off. We'll assume an initial mean state
#of 3 (i.e. we think we have 3 volts) and give the variance as 1
initial_guess = MvNormal([3.0], [1.0])
#End Section: Set initial guess of the state
################################################################################

################################################################################
#Section: Execute Kalman Filter
#-------------------------------------------------------------------------------
#Now that we have some noisy observations, the Kalman filter parameters and an
#intial guess for the state of the system, we can run the Kalman Filter
filtered_state = filter(linSSM, observations, initial_guess)
@printf("Log Likelihood: %.2f\n", filtered_state.loglik)
#End Section: Execute Kalman Filter
################################################################################

################################################################################
#Section: Plot Filtered results
#-------------------------------------------------------------------------------
#Here we are plotting the filtered results with Gadfly. See the Gadfly
#documentation for information about how plotting works if you are unfamiliar.
#Website: http://gadflyjl.org/

x_data = 1:number_of_observations
state_array = Vector{Float64}(number_of_observations+1)
confidence_array = Vector{Float64}(number_of_observations+1)
for i in 1:number_of_observations+1
    current_state = filtered_state.state[i]
    state_array[i] = current_state.μ[1]
    if i != 1
        confidence_array[i] = 2*sqrt(current_state.Σ.mat[1])
    else
        confidence_array[i] = 2*sqrt(current_state.Σ.diag[1])
    end
end
df_fs = DataFrame(
    x = [0;x_data],
    y = state_array,
    ymin = state_array - confidence_array,
    ymax = state_array + confidence_array,
    f = "Filtered values"
    )

n = 3
getColors = distinguishable_colors(n, Color[LCHab(70, 60, 240)],
                                   transform=c -> deuteranopic(c, 0.5),
                                   lchoices=Float64[65, 70, 75, 80],
                                   cchoices=Float64[0, 50, 60, 70],
                                   hchoices=linspace(0, 330, 24))
filtered_state_plot = plot(
    layer(x=x_data, y=filtered_state.observations, Geom.point, Theme(default_color=getColors[2])),
    layer(x=[0;x_data], y=ones(number_of_observations+1)*true_voltage, Geom.line, Theme(default_color=getColors[3])),
    layer(df_fs, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon),
    Guide.xlabel("Measurement Number"), Guide.ylabel("Voltage (Volts)"),
    Guide.manual_color_key("Colour Key",["Filtered Estimate", "Measurements","True Value "],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Linear Kalman Filter Example")
    )
display(filtered_state_plot)
#End Section: Plot Filtered results
################################################################################

################################################################################
#Section: Execute Kalman Smoother
#-------------------------------------------------------------------------------
#Everything can be performed much better in hindsight and that's essentially
#where the Kalman smoother comes in. Given all of the observations and the
#filtered state (including the Kalman filter parameters, the smoother attempts
#to give better estimates of the system's state. Here we perform smoothing on
#the filtered data.
smoothed_state = smooth(linSSM, filtered_state)
#End Section: Execute Kalman Filter
################################################################################

################################################################################
#Section: Plot Smoothed Results
#-------------------------------------------------------------------------------
#Here we are plotting the smoothed results with Gadfly. See the Gadfly
#documentation for information about how plotting works if you are unfamiliar.
#Website: http://gadflyjl.org/
state_array = Vector{Float64}(number_of_observations)
confidence_array = Vector{Float64}(number_of_observations)
for i in x_data
    current_state = smoothed_state.state[i]
    state_array[i] = current_state.μ[1]
    confidence_array[i] = 2*sqrt(current_state.Σ.mat[1])
end
df_ss = DataFrame(
    x = x_data,
    y = state_array,
    ymin = state_array - confidence_array,
    ymax = state_array + confidence_array,
    f = "Filtered values"
    )

smoothed_state_plot = plot(
    layer(x=x_data, y=smoothed_state.observations, Geom.point, Theme(default_color=getColors[2])),
    layer(x=x_data, y=ones(number_of_observations)*true_voltage, Geom.line, Theme(default_color=getColors[3])),
    layer(df_ss, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon),
    Guide.xlabel("Measurement Number"), Guide.ylabel("Voltage (Volts)"),
    Guide.manual_color_key("Colour Key",["Smoothed Estimate", "Measurements","True Value "],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Linear Kalman Smoother Example")
    )
display(smoothed_state_plot)
#End Section: Plot Filtered results
################################################################################
