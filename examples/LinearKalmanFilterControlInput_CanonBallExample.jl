#Linear Kalman filter with control input example
#This example closely follows that given on "Greg Czerniak's Website". Namely
#the canonball example on this page: http://greg.czerniak.info/guides/kalman1/
#Basically we shoot a canonball into the air at a given angle of elevation.
#Using observations of the canonball's velocity and position we're going to
#track the position of the canonball.

#To use the Kalman Filter we need three things:
#1) Some observations
#2) The parameters of the Kalman Filter model describing the system process and
#the observation model. In this example we also need the control input
#parameters.
#3) An initial guess of the state.
#

#Let's import the modules required to execute the Kalman Filter
using StateSpace
using Distributions
using DataFrames
using Gadfly
using Colors

################################################################################
#Section: Generate noisy Observations
#-------------------------------------------------------------------------------
#First we're going to provide some parameter values required to model the motion
#of the canonball.

#Set the Parameters
elevation_angle = 45.0 #Angle (measured anti-clockwise) between the ground (zero degrees) and the direction the canon ball is fired
muzzle_speed = 100.0 #Speed at which the canonball leaves the muzzle
initial_velocity = [muzzle_speed*cos(deg2rad(elevation_angle)), muzzle_speed*sin(deg2rad(elevation_angle))] #initial x and y components of the velocity
gravAcc = 9.81 #gravitational acceleration
initial_location = [0.0, 0.0] # initial position of the canonball
Δt = 0.1 #time between each measurement

#Functions describing the position of canonball
x_pos(x0::Float64, Vx::Float64, t::Float64) = x0 + Vx*t
y_pos(y0::Float64, Vy::Float64, t::Float64, g::Float64) = y0 + Vy*t - (g * t^2)/2
#Function to describe the evolution of the velocity in the vertical direction
velocityY(Vy::Float64, t::Float64, g::Float64) = Vy - g * t

#Give variances of the observation noise for the position and velocity
x_pos_var = 200.0
y_pos_var = 200.0
Vx_var = 1.0
Vy_var = 1.0

#Set the number of observations and preallocate vectors to store true and noisy
#measurement values
numObs = 145
x_pos_true = Vector{Float64}(numObs)
x_pos_obs = Vector{Float64}(numObs)
y_pos_true = Vector{Float64}(numObs)
y_pos_obs = Vector{Float64}(numObs)

Vx_true = Vector{Float64}(numObs)
Vx_obs = Vector{Float64}(numObs)
Vy_true = Vector{Float64}(numObs)
Vy_obs = Vector{Float64}(numObs)

#Generate the data (true values and noisy observations)
for i in 1:numObs
    x_pos_true[i] = x_pos(initial_location[1], initial_velocity[1], (i-1)*Δt)
    y_pos_true[i] = y_pos(initial_location[2], initial_velocity[2], (i-1)*Δt, gravAcc)
    Vx_true[i] = initial_velocity[1]
    Vy_true[i] = velocityY(initial_velocity[2], (i-1)*Δt, gravAcc)

    x_pos_obs[i] = x_pos_true[i] + randn() * sqrt(x_pos_var)
    y_pos_obs[i] = y_pos_true[i] + randn() * sqrt(y_pos_var)
    Vx_obs[i] = Vx_true[i] + randn() * sqrt(Vx_var)
    Vy_obs[i] = Vy_true[i] + randn() * sqrt(Vy_var)
end
#Create the observations vector for the Kalman filter
observations = [x_pos_obs Vx_obs y_pos_obs Vy_obs]'
#End Section: Generate noisy Observations
################################################################################

################################################################################
#Section: Describe Kalman Filter parameters
#-------------------------------------------------------------------------------

#Describe the system parameters
process_matrix = [[1.0, Δt, 0.0, 0.0] [0.0, 1.0, 0.0, 0.0] [0.0, 0.0, 1.0, Δt] [0.0, 0.0, 0.0, 1.0]]'
process_covariance = 0.01*eye(4)
observation_matrix = eye(4)
observation_covariance = 0.2*eye(4)
control_matrix = [[0.0, 0.0, 0.0, 0.0] [0.0, 0.0, 0.0, 0.0] [0.0, 0.0, 1.0, 0.0] [0.0, 0.0, 0.0, 1.0]]
control_input = [0.0, 0.0, -(gravAcc * Δt^2)/2, -(gravAcc * Δt)]

#Create an instance of the LKF with the control inputs
linCISMM = LinearGaussianCISSM(process_matrix, process_covariance, observation_matrix, observation_covariance, control_matrix, control_input)
#End Section: Describe Kalman Filter parameters
################################################################################

################################################################################
#Section: Set Initial Guess
#-------------------------------------------------------------------------------
initial_guess_state = [0.0, initial_velocity[1], 500.0, initial_velocity[2]]
initial_guess_covariance = eye(4)
initial_guess = MvNormal(initial_guess_state, initial_guess_covariance)
#End Section: Set Initial Guess
################################################################################

################################################################################
#Section: Execute Kalman Filter
#-------------------------------------------------------------------------------
filtered_state = filter(linCISMM, observations, initial_guess)
#End Section: Execute Kalman Filter
################################################################################

################################################################################
#Section: Plot Filtered results
#-------------------------------------------------------------------------------
#Here we are plotting the filtered results with Gadfly. See the Gadfly
#documentation for information about how plotting works if you are unfamiliar.
#Website: http://gadflyjl.org/
#Plot results
x_filt = Vector{Float64}(numObs)
y_filt = Vector{Float64}(numObs)
for i in 1:numObs
    current_state = filtered_state.state[i]
    x_filt[i] = current_state.μ[1]
    y_filt[i] = current_state.μ[3]
end

n = 3
getColors = distinguishable_colors(n, Color[LCHab(70, 60, 240)],
                                   transform=c -> deuteranopic(c, 0.5),
                                   lchoices=Float64[65, 70, 75, 80],
                                   cchoices=Float64[0, 50, 60, 70],
                                   hchoices=linspace(0, 330, 24))

cannonball_plot = plot(
    layer(x=x_pos_true, y=y_pos_true, Geom.line, Theme(default_color=getColors[3])),
    layer(x=[initial_guess_state[1]; x_filt], y=[initial_guess_state[3]; y_filt], Geom.line, Theme(default_color=getColors[1])),
    layer(x=x_pos_obs, y=y_pos_obs, Geom.point, Theme(default_color=getColors[2])),
    Guide.xlabel("X position"), Guide.ylabel("Y position"),
    Guide.manual_color_key("Colour Key",["Filtered Estimate", "Measurements","True Value "],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Measurement of a Canonball in Flight")
    )
#End Section: Plot Filtered results
################################################################################