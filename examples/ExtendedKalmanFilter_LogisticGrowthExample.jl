#An Extended Kalman Filter (EKF) example taken from Markus Gesmann's
#Mages' blog titled "Extended Kalman filter example in R" [1] which in turn
#was taken from a blog post titled "Fun with (Extended Kalman) Filters"
#by Dominic Steinitz [2].
#The web addresses are:
#[1] http://www.magesblog.com/2015/01/extended-kalman-filter-example-in-r.html
#[2] https://idontgetoutmuch.wordpress.com/2014/09/09/fun-with-extended-kalman-filters-4/

#Here we import the required modules
using StateSpace
using Distributions
using Gadfly
using DataFrames
using Colors

################################################################################
#Section: Generate noisy Observations
#-------------------------------------------------------------------------------

#Here we set the parameters with their true values
r = 0.2 #r is the growth rate
k = 100.0 #k is the carrying capacity
p0 = 0.1 * k # p0 is the initial population
Δt = 0.1 # Δt is the change in time.

#Define the logistic growth function to set the observations
function logisticGrowth(r, p, k, t)
    k * p * exp(r*t) / (k + p * (exp(r*t) - 1))
end
logisticGrowth(state) = logisticGrowth(state[1], state[2], k, Δt)

#Set the measurement noise variance
measurement_noise_variance = 25.0

#create the noisy observations (zero mean Gaussian noise)
numObs = 100
true_values = Vector{Float64}(numObs)
population_measurements = Vector{Float64}(numObs)
for i in 1:numObs
    true_values[i] = logisticGrowth(r, p0, k, i*Δt)
    population_measurements[i] = true_values[i] + randn() * sqrt(measurement_noise_variance)
end

#Since we're going to assume that our state consists of a growth rate, r,
#as well as a population rate, p, we need measurements of the growth rate.
#The problem is that we don't actually observe the growth rate directly.
#So we will set these measurements to zero.
growth_rate_measurements = zeros(numObs)

#Then we put these measurements together
measurements = [growth_rate_measurements population_measurements]'
#End Section: Generate noisy Observations
################################################################################

################################################################################
#Section: Describe Extended Kalman Filter parameters
#-------------------------------------------------------------------------------
#The state consists of the growth rate, r, and the population number, p. We'll
#assume that the growth rate is constant throughout time. And we'll assume that
#the population update follows the logistic growth pattern. So let's create that
#process function
function process_fcn(state)
    predict_growth_rate = state[1]
    predict_population = logisticGrowth(state)
    new_state = [predict_growth_rate, predict_population]
    return new_state
end
#Here we assume that there is no evolution noise so well create a zero matrix
process_noise_mat = diagm([0.001, 0.001])

#Now we need to describe the observation model. We'll assume that we don't
#observe the growth rate but we do observe the population. So the observation
#function is:
function observation_fcn(state)
    growth_rate_observation = 0.0
    population_observation = 1.0 * state[2]
    observation = [growth_rate_observation, population_observation]
    return observation
end

#Now we need to set the observation noise. We already set the measurement noise
#for the population earlier. We'lll arbitrarily make the variance for the
#growth rate (it doesn't matter because we don't explicity observe it):
observation_noise_mat = diagm([1.0, measurement_noise_variance])

#Create instance of our EKF model
nonLinSSM = NonlinearGaussianSSM(process_fcn, process_noise_mat, observation_fcn, observation_noise_mat)
#End Section: Describe Extended Kalman Filter parameters
################################################################################

################################################################################
#Section: Set initial guess of the state
#-------------------------------------------------------------------------------
initial_guess = MvNormal([0.5, 10], diagm([1.0,20.0]))
################################################################################


################################################################################
#Section: Execute the Extended Kalman Filter
#-------------------------------------------------------------------------------
filtered_state = filter(nonLinSSM, measurements, initial_guess)
#End Section: Execute  the Extended Kalman Filter
################################################################################


################################################################################
#Section: Plot Filtered results
#-------------------------------------------------------------------------------
#Here we are plotting the filtered results with Gadfly. See the Gadfly
#documentation for information about how plotting works if you are unfamiliar.
#Website: http://gadflyjl.org/

x_data = 1:numObs
population_array = Vector{Float64}(numObs+1)
confidence_array = Vector{Float64}(numObs+1)
population_array[1] = initial_guess.μ[2]
confidence_array[1] = 2*sqrt(initial_guess.Σ.mat[2,2])
for i in x_data
    current_state = filtered_state.state[i]
    population_array[i+1] = current_state.μ[2]
    confidence_array[i+1] = 2*sqrt(current_state.Σ.mat[2,2])
end
df_fs = DataFrame(
    x = [0;x_data],
    y = population_array,
    ymin = population_array - confidence_array,
    ymax = population_array + confidence_array,
    f = "Filtered values"
    )

n = 3
getColors = distinguishable_colors(n, Color[LCHab(70, 60, 240)],
                                   transform=c -> deuteranopic(c, 0.5),
                                   lchoices=Float64[65, 70, 75, 80],
                                   cchoices=Float64[0, 50, 60, 70],
                                   hchoices=linspace(0, 330, 24))
population_state_plot = plot(
    layer(x=0:numObs, y=[p0;measurements[2,:]'], Geom.point, Theme(default_color=getColors[2])),
    layer(x=0:numObs, y=[p0;true_values], Geom.line, Theme(default_color=getColors[3])),
    layer(df_fs, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon),
    Guide.xlabel("Measurement Number"), Guide.ylabel("Population"),
    Guide.manual_color_key("Colour Key",["Filtered Estimate", "Measurements","True Value "],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Extended Kalman Filter Example")
    )
display(population_state_plot)
#End Section: Plot Filtered results
################################################################################


################################################################################
#Section: Plot Filtered results
#-------------------------------------------------------------------------------
#Here we are plotting the filtered results with Gadfly. See the Gadfly
#documentation for information about how plotting works if you are unfamiliar.
#Website: http://gadflyjl.org/

x_data = 1:numObs
growth_rate_array = Vector{Float64}(numObs+1)
confidence_array = Vector{Float64}(numObs+1)
growth_rate_array[1] = initial_guess.μ[1]
confidence_array[1] = initial_guess.Σ.mat[1,1]
for i in x_data
    current_state = filtered_state.state[i]
    growth_rate_array[i+1] = current_state.μ[1]
    confidence_array[i+1] = 2*sqrt(current_state.Σ.mat[1,1])
end
df_fs = DataFrame(
    x = [0;x_data],
    y = growth_rate_array,
    ymin = growth_rate_array - confidence_array,
    ymax = growth_rate_array + confidence_array,
    f = "Filtered values"
    )

growth_rate_state_plot = plot(
    layer(x=0:numObs, y=ones(numObs+1)*r, Geom.line, Theme(default_color=getColors[3])),
    layer(df_fs, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon),
    Guide.xlabel("Measurement Number"), Guide.ylabel("Growth Rate"),
    Guide.manual_color_key("Colour Key",["Filtered Estimate", "Measurements","True Value "],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Extended Kalman Filter Example")
    )
display(growth_rate_state_plot)
#End Section: Plot Filtered results
################################################################################
