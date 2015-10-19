#Missing observations example
#This example is a really basic/unrealistic example that was made up
#to demonstrate the use of missing observations with the StateSpace.jl
#package. In this example we have two (spherical) balloons that deflate
#over time. We record (noisy) measurements of the radius of only one of
#the balloons over time. We assume the process is that the balloon's
#radius decreases by 10% of it's previous radius with each observation.
#We consider 3 cases for the unobserved balloon:
# 1) Our best guess is the prior prediction
# 2) We have some information on the measurement error
# 3) We know that there is some correlation between both balloons.

#Let's import the modules required to execute the Kalman Filter and visualize
#the results
using StateSpace
using Distributions
using Gadfly
using DataFrames
using Colors

################################################################################
#Section: Set initial guess of the state
#-------------------------------------------------------------------------------
#We'll begin with a guess and set the variances to 1 as default. Notice that the
#radius of the first balloon is assumed to be twice that of the second balloon.
#I've included a third balloon here since it will make the effects of the 3rd
#case (mentioned above) clearer
initialState = [30.0, 15.0, 15.0]

initialCov = [1.0 0.0 0.0;
               0.0 1.0 0.0;
               0.0 0.0 1.0]
initialGuess = MvNormal(initialState, initialCov)
#End Section: Set initial guess of the state
################################################################################

################################################################################
#Section: Generate noisy Observations
#-------------------------------------------------------------------------------
#Here we generate the noisy observations of each balloon's radius
numObs = 30
numBalloons = length(initialState)
observationVariance = 2.0
observations = Matrix{Float64}(numBalloons,numObs)
for i in 1:numObs
    if i == 1
        observations[1,1] = 0.9 * initialState[1] + randn()*sqrt(observationVariance)
    else
        observations[1,i] = 0.9 * observations[1,i-1] + randn()*sqrt(observationVariance)
    end
    observations[2:end,i] = [NaN, NaN]
end
#End Section: Generate noisy Observations
################################################################################

################################################################################
#Section: Describe Kalman Filter parameters
#-------------------------------------------------------------------------------
#The process is the deflation of the balloons over time. We'll assume that they
#deflate by 10% of their previous state for each iteration and again set the
#variances to 1.
processMatrix = 0.9 * eye(initialCov)
processCov = 1.0 * eye(initialCov)

#We'll assume that we measure the radius directly and hence the observation
#matrix is equivalent to the identity matrix. The measurement covariance matrix
#will be described in terms of a correlation matrix (i.e. correlation values
#between -1 and 1 corresponding to perfect anticorrelation and perfect
#correlation respectively). This means that we'll need a function that will
#convert a correlation matrix to a covariance matrix.

function corr2covar{T}(correlationMatrix::Matrix{T}, sigmaVector::Vector{T})
    N = length(sigmaVector)
    covarianceMatrix = zeros(correlationMatrix)
    for j in 1:N
        for i in 1:N
            covarianceMatrix[i, j] = correlationMatrix[i,j] * sigmaVector[i] * sigmaVector[j]
        end
    end
    return covarianceMatrix
end

observationMatrix = eye(processMatrix) #Define observation matrix

#Here we define the correlation matrix to describe the correlations bewteen
#each balloon. In this matrix, we assume the balloon measurements are
#uncorrelated.
observationCorr = [1.0 0.0 0.0;
                   0.0 1.0 0.0;
                   0.0 0.0 1.0]
#Now we define a vector containing standard deviation values for the measurment
#variances
observationΣ = [1.0, 3.0, 3.0]
observationCov = corr2covar(observationCorr, observationΣ) #convert correlation matrix to covariance matrix

#Create the linear StateSpace model.
linSSM = LinearGaussianSSM(processMatrix, processCov,
                           observationMatrix, observationCov)
#End Section: Describe Kalman Filter parameters
################################################################################

################################################################################
#Section: Perform Kalman Filter for cases 1 and 2
#-------------------------------------------------------------------------------
#Case 1: Our best guess is the prior prediction
#If we're completely uncertain of the measurement variances then we update the
#state with our predicted state. The error in doing this will propagte and we'll
#have an estimate with a large uncertainty. The syntax for this case is the same
#as the usual syntax for the package since this is the default behaviour (and
#the safest, although not necessarily optimal).
filtState_c1 = filter(linSSM, observations, initialGuess)

#Case 2: We have some information on the measurement error
#If we think that we have some information about the measurement errors then we
#can take advantage of that by using it on our predicted observersation (i.e.
#using our measurement errors on our predicted observations that are generated
#by applying the observation model to the state that is predicted from the
#process model). The syntax for this requires an extra parameter for the filter
#method - 'true'. This tells the filter that we are happy to use our measurement
#error estimates.
filtState_c2 = filter(linSSM, observations, initialGuess, true)
#End Section: Perform Kalman Filter for cases 1 and 2
################################################################################

################################################################################
#Section: Perform Kalman Filter for case 3
#-------------------------------------------------------------------------------
#Case 3: We know that there is some correlation between balloons 1 and 2.
#For this case we assume that there is some correlation between the measurement
#errors. So we need to redefine the observation covariance matrix
observationCorr = [1.0 0.6 0.0;
                   0.6 1.0 0.0;
                   0.0 0.0 1.0]
observationCov = corr2covar(observationCorr, observationΣ) #convert correlation matrix to covariance matrix
#Create the linear StateSpace model with new observation covariance matrix.
linSSM2 = LinearGaussianSSM(processMatrix, processCov,
                           observationMatrix, observationCov)

#Now we can perform the Kalman filter with the correlated observation errors.
#Notice that the syntax is exactly the same as before.
filtState_c3 = filter(linSSM2, observations, initialGuess, true)
#End Section: Perform Kalman Filter for case 3
################################################################################

################################################################################
#Section: Plot Filtered results
#-------------------------------------------------------------------------------
#Balloon radius and 95% confidence for each balloon - case 1
b1_c1 = Vector{Float64}(numObs+1)
b2_c1 = Vector{Float64}(numObs+1)
b3_c1 = Vector{Float64}(numObs+1)
b1Sig_c1 = Vector{Float64}(numObs+1)
b2Sig_c1 = Vector{Float64}(numObs+1)
b3Sig_c1 = Vector{Float64}(numObs+1)

#Balloon radius and 95% confidence for each balloon - case 2
b1_c2 = Vector{Float64}(numObs+1)
b2_c2 = Vector{Float64}(numObs+1)
b3_c2 = Vector{Float64}(numObs+1)
b1Sig_c2 = Vector{Float64}(numObs+1)
b2Sig_c2 = Vector{Float64}(numObs+1)
b3Sig_c2 = Vector{Float64}(numObs+1)

#Balloon radius and 95% confidence for each balloon - case 3
b1_c3 = Vector{Float64}(numObs+1)
b2_c3 = Vector{Float64}(numObs+1)
b3_c3 = Vector{Float64}(numObs+1)
b1Sig_c3 = Vector{Float64}(numObs+1)
b2Sig_c3 = Vector{Float64}(numObs+1)
b3Sig_c3 = Vector{Float64}(numObs+1)

#Extract state estimates from each balloon for each case
for i in 1:numObs+1
    currentState1 = filtState_c1.state[i]
    b1_c1[i] = currentState1.μ[1]
    b2_c1[i] = currentState1.μ[2]
    b3_c1[i] = currentState1.μ[3]
    b1Sig_c1[i] = 2*sqrt(currentState1.Σ.mat[1,1])
    b2Sig_c1[i] = 2*sqrt(currentState1.Σ.mat[2,2])
    b3Sig_c1[i] = 2*sqrt(currentState1.Σ.mat[3,3])

    currentState2 = filtState_c2.state[i]
    b1_c2[i] = currentState2.μ[1]
    b2_c2[i] = currentState2.μ[2]
    b3_c2[i] = currentState2.μ[3]
    b1Sig_c2[i] = 2*sqrt(currentState2.Σ.mat[1,1])
    b2Sig_c2[i] = 2*sqrt(currentState2.Σ.mat[2,2])
    b3Sig_c2[i] = 2*sqrt(currentState2.Σ.mat[3,3])

    currentState3 = filtState_c3.state[i]
    b1_c3[i] = currentState3.μ[1]
    b2_c3[i] = currentState3.μ[2]
    b3_c3[i] = currentState3.μ[3]
    b1Sig_c3[i] = 2*sqrt(currentState3.Σ.mat[1,1])
    b2Sig_c3[i] = 2*sqrt(currentState3.Σ.mat[2,2])
    b3Sig_c3[i] = 2*sqrt(currentState3.Σ.mat[3,3])
end

#Create data frames for each balloon for each case
df_b1_c1 = DataFrame(
    x = 1:numObs+1,
    y = b1_c1,
    ymin = b1_c1 - b1Sig_c1,
    ymax = b1_c1 + b1Sig_c1,
    f = "Filtered values"
    )

df_b2_c1 = DataFrame(
    x = 1:numObs+1,
    y = b2_c1,
    ymin = b2_c1 - b2Sig_c1,
    ymax = b2_c1 + b2Sig_c1,
    f = "Filtered values"
    )

df_b3_c1 = DataFrame(
    x = 1:numObs+1,
    y = b3_c1,
    ymin = b3_c1 - b3Sig_c1,
    ymax = b3_c1 + b3Sig_c1,
    f = "Filtered values"
    )

df_b1_c2 = DataFrame(
    x = 1:numObs+1,
    y = b1_c2,
    ymin = b1_c2 - b1Sig_c2,
    ymax = b1_c2 + b1Sig_c2,
    f = "Filtered values"
    )

df_b2_c2 = DataFrame(
    x = 1:numObs+1,
    y = b2_c2,
    ymin = b2_c2 - b2Sig_c2,
    ymax = b2_c2 + b2Sig_c2,
    f = "Filtered values"
    )

df_b3_c2 = DataFrame(
    x = 1:numObs+1,
    y = b3_c2,
    ymin = b3_c2 - b3Sig_c2,
    ymax = b3_c2 + b3Sig_c2,
    f = "Filtered values"
    )

df_b1_c3 = DataFrame(
    x = 1:numObs+1,
    y = b1_c3,
    ymin = b1_c3 - b1Sig_c3,
    ymax = b1_c3 + b1Sig_c3,
    f = "Filtered values"
    )

df_b2_c3 = DataFrame(
    x = 1:numObs+1,
    y = b2_c3,
    ymin = b2_c3 - b2Sig_c3,
    ymax = b2_c3 + b2Sig_c3,
    f = "Filtered values"
    )

df_b3_c3 = DataFrame(
    x = 1:numObs+1,
    y = b3_c3,
    ymin = b3_c3 - b3Sig_c3,
    ymax = b3_c3 + b3Sig_c3,
    f = "Filtered values"
    )

#Get the correct colours for the plots
n = 3
getColors = distinguishable_colors(n, Color[LCHab(70, 60, 240)],
                                   transform=c -> deuteranopic(c, 0.5),
                                   lchoices=Float64[65, 70, 75, 80],
                                   cchoices=Float64[0, 50, 60, 70],
                                   hchoices=linspace(0, 330, 24))

#Plot the balloon radius estimates for each case
plt_c1 = plot(
    layer(x=2:numObs+1, y=observations[1,:], Geom.point, Theme(default_color=getColors[1])),
    layer(df_b1_c1, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[1])),
    layer(df_b2_c1, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[2])),
    layer(df_b3_c1, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[3])),
    Guide.xlabel("Measurement Number"), Guide.ylabel("Balloon Radius (cm)"),
    Guide.manual_color_key("Colour Key",["Balloon 1", "Balloon 2","Balloon 3"],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Missing Observations Example - Case 1")
    )
display(plt_c1)

plt_c2 = plot(
    layer(x=2:numObs+1, y=observations[1,:], Geom.point, Theme(default_color=getColors[1])),
    layer(df_b1_c2, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[1])),
    layer(df_b2_c2, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[2])),
    layer(df_b3_c2, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[3])),
    Guide.xlabel("Measurement Number"), Guide.ylabel("Balloon Radius (cm)"),
    Guide.manual_color_key("Colour Key",["Balloon 1", "Balloon 2","Balloon 3"],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Missing Observations Example - Case 2")
    )
display(plt_c2)

plt_c3 = plot(
    layer(x=2:numObs+1, y=observations[1,:], Geom.point, Theme(default_color=getColors[1])),
    layer(df_b1_c3, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[1])),
    layer(df_b2_c3, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[2])),
    layer(df_b3_c3, x=:x, y=:y, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, Theme(default_color=getColors[3])),
    Guide.xlabel("Measurement Number"), Guide.ylabel("Balloon Radius (cm)"),
    Guide.manual_color_key("Colour Key",["Balloon 1", "Balloon 2","Balloon 3"],[getColors[1],getColors[2],getColors[3]]),
    Guide.title("Missing Observations Example - Case 3")
    )
display(plt_c3)
#End Section: Plot Filtered results
################################################################################
