# StateSpace.jl

[![Build Status](https://travis-ci.org/ElOceanografo/StateSpace.jl.svg?branch=master)](https://travis-ci.org/ElOceanografo/StateSpace.jl)

## A Julia package for state space modeling

State space models are a very general type of dynamic statistical model, and have been used to estimate everything from biological populations to the position of Apollo 11 to the weather this weekend.  In a nutshell, they are useful when we want to know the *state* of some process, but we can't observe it directly.  They have two main pieces.  First is the *process model*, which describes probabilistically how the hidden state evolves from one time step to the next.  Second is the *observation model*, which describes, again probabilistically, how the state is translated into the quantities we observe.

These process and observation functions can be linear or nonlinear, and the process noise and observation errors may be Gaussian, or from some other probability distribution.  This package aims to provide methods to perform the common prediction, filtering, and smoothing tasks for each type of model.


### State estimates and predictions as probability distributions
Julia's type system facilitates an unconventional approach to representing the state estimates.  In most implementations of the Kalman and related filters, the state is tracked as a mean vector, with the covariance served "on the side." This package, on the other hand, tracks the state as a *distribution*, using the excellent [Distributions](https://github.com/JuliaStats/Distributions.jl) package.

There are several advantages to this approach.  The code is made shorter and clearer.  Expected values and confidence intervals can be calculated easily using the `mean` and `quantile` methods defined for distributions.  It is easy to generate random draws from state estimates, for instance if you wanted to bootstrap some quantity derived from the state.  Finally, it is just more elegant, and closer to the true meaning of the statistical model.

### Modeling interface
There are three basic tasks in state space modeling.  All types of models define the same methods to do them.

#### Prediction
Given a current estimate of the state, where do we expect it to be at the next time step, and how certain are we?  

* `predict(model, state)` 

#### Updating/filtering
Given a predicted state and a new observation, how do we combine the prediction with the new data to get a better (or optimal) estimate of the state?  This can be done either "on line," i.e. one step at a time as each new measurement arrives, or all at once after all the data have been collected.

`update(model, predicted_state, data)`
`update!(model, filtered_states, data)`
`filter(model, data, state0)`

#### Smoothing
Given a full set of filtered state estimates and a full data set, go back and revise all the estimates to their optimum value, given all the information.  Hindsight is 20/20!
`smooth(filtered_state)`


### Algorithms already implemented:
* The Kalman[[1]](1) filter.  This is a fast and optimal technique for linear systems with Gaussian process noise and observational error.

* The extended Kalman filter.  This method can be applied to nonlinear problems with Gaussian noise and error.  It works by linearizing the equations around the current state estimate, calculating the required Jacobians automagically using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

### Algorithms on the way:
* The unscented[[2]](2) Kalman filter.  Applicable to nonlinear and non-Gaussian problems.
* The particle filter.  Applicable to nonlinear and non-gaussian problems.

### Model fitting
Currently not implemented in this package, though it may be in the future.  `FilteredState` objects contain the model's log-likelihood, which can be used to optimize parameter values.  

[1] Named for [Rudolph E. Kalman](http://en.wikipedia.org/wiki/Rudolf_E._K%C3%A1lm%C3%A1n).
[2] Named for a stick of unscented deodorant.  [No, really](http://www.ieeeghn.org/wiki/index.php/First-Hand:The_Unscented_Transform#What.E2.80.99s_with_the_Name_.E2.80.9CUnscented.E2.80.9D.3F).