State Space Examples List
=========================

###Introduction

The *examples* directory contains scripts and IJulia notebooks for each Kalman Filter example. The **scripts can be run as they are** to produce the results and are fully commented as standalone examples. The IJulia notebook for the corresponding example exists to provide a bit more context to the problem and guide the user through the problem setup and how to solve the problem using the StateSpace.jl package.

###Examples

#####Basic Linear Kalman Filter and smoother - Voltage Example

The [script](https://github.com/JonnyCBB/StateSpace.jl/blob/master/examples/LinearKalmanFilter_VoltageExample.jl) and the [notebook](https://github.com/JonnyCBB/StateSpace.jl/blob/master/examples/LinearKalmanFilter_VoltageExample.ipynb) describe a problem where a constant true voltage is measured with a noisy voltmeter. The results of the filter can be seen here: ![LKF filter](figures/LKF_filtered_plot.png)
