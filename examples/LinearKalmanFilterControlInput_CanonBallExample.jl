#Let's import the modules required to execute the Kalman Filter
using StateSpace
using Distributions
using DataFrames
using Gadfly
using Colors

#Parameters
elevation_angle = 45.0
muzzle_speed = 100.0
initial_velocity = [muzzle_speed*cos(deg2rad(elevation_angle)), muzzle_speed*sin(deg2rad(elevation_angle))]
gravAcc = 9.81
initial_location = [0.0, 0.0]
Δt = 0.1

#Functions describing the position of canonball
x_pos(x0::Float64, Vx::Float64, t::Float64) = x0 + Vx*t
x_pos(state) = x_pos(state[1], state[2], Δt)

y_pos(y0::Float64, Vy::Float64, t::Float64, g::Float64) = y0 + Vy*t - (g * t^2)/2
y_pos(state) = y_pos(state[1], state[2], Δt, gravAcc)

velocityY(Vy::Float64, t::Float64, g::Float64) = Vy - g * t

#generate noisy observations
x_pos_var = 200.0
y_pos_var = 200.0
Vx_var = 1.0
Vy_var = 1.0

numObs = 145
x_pos_true = Vector{Float64}(numObs)
x_pos_obs = Vector{Float64}(numObs)
y_pos_true = Vector{Float64}(numObs)
y_pos_obs = Vector{Float64}(numObs)

Vx_true = Vector{Float64}(numObs)
Vx_obs = Vector{Float64}(numObs)
Vy_true = Vector{Float64}(numObs)
Vy_obs = Vector{Float64}(numObs)
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
observations = [x_pos_obs Vx_obs y_pos_obs Vy_obs]'

#Describe the system parameters
process_matrix = [[1.0, Δt, 0.0, 0.0] [0.0, 1.0, 0.0, 0.0] [0.0, 0.0, 1.0, Δt] [0.0, 0.0, 0.0, 1.0]]'
process_covariance = 0.01*eye(4)
observation_matrix = eye(4)
observation_covariance = 0.2*eye(4)
control_matrix = [[0.0, 0.0, 0.0, 0.0] [0.0, 0.0, 0.0, 0.0] [0.0, 0.0, 1.0, 0.0] [0.0, 0.0, 0.0, 1.0]]
control_input = [0.0, 0.0, -(gravAcc * Δt^2)/2, -(gravAcc * Δt)]

linCISMM = LinearGaussianCISSM(process_matrix, process_covariance, observation_matrix, observation_covariance, control_matrix, control_input)

#Initial Guess
initial_guess_state = [0.0, initial_velocity[1], 500.0, initial_velocity[2]]
initial_guess_covariance = eye(4)
initial_guess = MvNormal(initial_guess_state, initial_guess_covariance)

#filter results
filtered_state = filter(linCISMM, observations, initial_guess)

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