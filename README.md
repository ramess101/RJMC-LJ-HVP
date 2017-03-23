# RJMC-LJ-HVP
Reversible Jump MCMC for the Lennard-Jones fluid when predicting Heat of Vaporization

The main code is "RJMC_LJ_HVP.py". This file:

1. Extracts values for argon from "argon.yaml"
2. Reads in the experimental data from "HVP_data_argon.csv"
3. Imports the "LJ_fluid_correlations.py" functions which use the parameters found in "LJ_fluid.yaml"
4. Contains two primary modules:
  a) Calc_posterior which calculates the posterior distribution for a given epsilon and sigma
  b) RJMC_tuned which changes the values of epsilon and sigma according to a random walk Markov Chain
  
The unique aspect of this code is that it considers two models: one parameter and two parameters. The one parameter model only uses the LJ parameter epsilon while the two parameter model varies both epsilon and sigma. Since Heat of Vaporization only depends on epsilon the expected result is that the one parameter model will be favored for its simplicity over the two parameter model.

Both models have been given an equal probability and the Jacobian has a value of 1 since the value of sigma does not change when converting between the models.

The results that are typically obtained from running this code are:

Acceptance Rate during production for eps, sig, t: [ 0.31  0.32   0.37]

Acceptance model swap during production: [ 0.24]

Percent that 1-parameter model is sampled: 75.4

Typical plots of the results are found as PDF files.

  argon_Model_Params_RJMC.pdf depicts the number of parameters in the model for each iteration and a histogram comparing the prevalency of the two models
  
  argon_Prop_RJMC.pdf presents the uncertainty in HVP according to the predictive posterior
  
  argon_Trace_RJMC.pdf depicts the trace for epsilon, sigma, and t (the precision, i.e. 1/variance)
  
  argon_Trajectory_RJMC.pdf depicts the trajectory through the parameter space of accepted moves

The primary questions are:

1. Why is the 1-parameter model only sampled 75% of the time? Should this value not be higher? Specifically, this value yields a Bayes Factor of 3 for the 1-parameter model which is "barely worth mentioning" level of evidence.
2. How can we improve the algorithm? For example, the frequency of model swaps, the priors, etc.
3. Is our acceptance criterion rigorous?
