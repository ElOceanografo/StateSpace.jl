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

################################################################################
#Section: Generate noisy Observations
#-------------------------------------------------------------------------------

#Here we set the parameters with their true values
r = 0.2 #r is the growth rate
k = 100.0 #k is the carrying capacity
p0 = 0.1 * k # p0 is the initial population
Δt = 0.1 # Δt is the change in time.

#Define the logistic growth function to set the observations
function logisticGrowth(r::Float64, p::Float64, k::Float64, t::Float64)
    k * p * exp(r*t) / (k + p * (exp(r*t) - 1))
end
logisticGrowth(state) = logisticGrowth(state[1], state[2], k, Δt)

#Set the measurement noise variance
measurement_noise_variance = 25

#create the noisy observations (zero mean Gaussian noise)
numObs = 250
true_values = Vector{Float64}(numObs)
measurements = Vector{Float64}(numObs)
for i in 1:numObs
    true_values[i] = logisticGrowth(r, p0, k, i*Δt)
    measurements[i] = true_values[i] + randn() * sqrt(measurement_noise_variance)
end
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
process_noise_mat = zeros(2,2)

#Now we need to describe the observation model. We'll assume that we don't
#observe the growth rate but we do observe the population. So the observation
#function is:
function observation_fcn(state::Vector{Float64})
    growth_rate_observation = 0.0
    population_observation = 1.0
    observation = [growth_rate_observation, population_observation]
    return observation
end

#Now we need to set the observation noise. Since we don't observe the growth
#rate, the variance for the growth rate is zero. We already set the
#measurement noise earlier. So we have:
observation_noise_mat = Matrix(Diagonal([0.0, measurement_noise_variance]))

#Create instance of our EKF model
nonLinSSM = NonlinearGaussianSSM(process_fcn, process_noise_mat, observation_fcn, observation_noise_mat)
#End Section: Describe Extended Kalman Filter parameters
################################################################################

################################################################################
#Section: Set initial guess of the state
#-------------------------------------------------------------------------------
initial_guess = MvNormal([0.2], [10.0])
################################################################################


################################################################################
#Section: Execute  the Extended Kalman Filter
#-------------------------------------------------------------------------------
filtered_state = filter(measurements, nonLinSSM, initial_guess)
#End Section: Execute  the Extended Kalman Filter
################################################################################

# plot(layer(x=0:numObs, y=[p0;measurements], Geom.point),
#      layer(x=0:numObs, y=[p0;true_values], Geom.line))
