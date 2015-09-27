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

#Define the logistic growth function
function logisticGrowth(r::Float64, p::Float64, k::Float64, t::Float64)
    k * p * exp(r*t) / (k + p * (exp(r*t) - 1))
end

#Here we set the parameters with their true values
r = 0.2 #r is the growth rate
k = 100.0 #k is the carrying capacity
p0 = 0.1 * k # p0 is the initial population
Δt = 0.1 # Δt is the change in time.

#Set the measurement noise variance
measurement_noise_variance = 25

#create the noisy observations
numObs = 250
true_values = Vector{Float64}(numObs)
measurements = Vector{Float64}(numObs)
for i in 1:numObs
    measurements[i] = logisticGrowth(r, p0, k, i*Δt) + randn() * sqrt(measurement_noise_variance)
end

plot(x=0:numObs, y=[p0;measurements], Geom.line)