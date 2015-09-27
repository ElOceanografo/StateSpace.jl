
* All of these tasks should work for univariate and multivariate states.

* Methods should be defined for every filtering algorithm.

PredictedState
FilteredState
SmoothedState

#### Prediction ####
Given a current estimate of the state, where do we expect it to be at the next time step, and how certain are we?  

* `predict(m<:AbtractSSM, x<:Distribution)` returns `x1 <: Distribution`
* `predict(m<:AbtractSSM, fs<:FilteredState)` returns `x1 <: Distribution`

Should a predict method be defined for point estimates of a state (i.e., a vector instead of a distribution)?

#### Updating ####
Given a predicted state and a new observation, how do we combine the prediction with the new data to get a better (or optimal) estimate of the state?  This can be done either "on line," i.e. one step at a time as each new measurement arrives, or all at once after all the data have been collected.

* `update(m<:AbstractSSM, pred<:Distribution, data<:Vector)`

* `update!(m<:AbstractSSM, fs<:FilteredState, data<:Vector)`

#### Filtering ####
"Batch mode" updating

#### Smoothing ####
Given a full set of filtered state estimates and a full data set, go back and revise all the estimates to their optimum value, given all the information.  Hindsight is 20/20!

* `smooth(filtered_state)`

#### Model Fitting ####