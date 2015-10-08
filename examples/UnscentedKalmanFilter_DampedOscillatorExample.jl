# Example from http://folk.ntnu.no/bjarnean/pubs/journals/journal-47.pdf

#Here we import the required modules
using StateSpace
using Distributions
using Gadfly
using DataFrames
using Colors

#######################################
Δt = 0.1
function f(x::Vector, Δt::Float64=0.1, μ::Float64=0.3)
    x1 = zeros(x)
    x1[1] = x[1] + Δt * -x[2]
    x1[2] = x[2] + Δt * (-μ * (1 - x[1]^2) * x[2] + x[1])
    x1
end
h(x::Vector) = x
R = 1e-2*[0.1 0;
    0 1e-3]
Q = 1e-1*eye(2)
m = AdditiveNonLinUKFSSM(f, R, h, Q)

#######################
trueInitialState = [1.4, 0.0]

numObs = 200
trueState = zeros(2,numObs)
noisyMeas = zeros(2,numObs)
trueState[:,1] = trueInitialState
noisyMeas[:,1] = trueInitialState + sqrt(Q)*randn(2)
for i in 2:numObs
    trueState[:,i] = f(trueState[:,i-1])
    noisyMeas[:,i] = trueState[:,i] + sqrt(Q)*randn(2)
end
# plt = plot(layer(x=1:numObs, y=trueState[1,:], Geom.line),
#      layer(x=1:numObs, y=noisyMeas[1,:], Geom.point))
# display(plt)
###########################################################

initial_guess = MvNormal([0.0,5.0], 5.0*eye(2))

###########################################################

filtered_state = filter(m, noisyMeas, initial_guess)

###########################################################

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
population_state_plot = plot(
    layer(x=x_data*Δt, y=noisyMeas[1,:], Geom.point, Theme(default_color=getColors[2])),
    layer(x=x_data*Δt, y=trueState[1,:], Geom.line, Theme(default_color=getColors[3])),
    layer(df_fs, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon),
    Guide.xlabel("Time (seconds)"), Guide.ylabel("x1"),
    Guide.manual_color_key("Colour Key",["Filtered Estimate", "Measurements","True Value "],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Unscented Kalman Filter (Additive) Example")
    )
display(population_state_plot)
