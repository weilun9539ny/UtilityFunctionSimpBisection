// Vectorized Hierarchical Bayesian Model for the CRRA Utility Function
// This version uses pre-split data (positive/negative x) and
// Non-Centered Parameterization (NCP) for faster, more robust sampling.

data {
  // --- General ---
  int<lower=1> J; // Number of subjects
  int<lower=1> N; // Number of Observation (separated in positive and negative data)
  array[N] int<lower=1, upper=J> N_subj; // Subject index

  // --- Positive Data (x >= 0) ---
  // int<lower=0> N_pos; // Number of observations where x >= 0
  vector[N] x_pos;
  vector[N] y_pos;

  // --- Negative Data (x < 0) ---
  // int<lower=0> N_neg; // Number of observations where x < 0
  vector[N] x_neg; // Since pow() takes positive value, input x_neg should be abs()
  vector[N] y_neg;
}

parameters {
  // --- Population-level parameters (Hyperparameters) ---
  real<lower=0> mu_alpha; // Population mean for alpha
  real<lower=0> mu_beta;  // Population mean for beta
  real<lower=0> tau_alpha; // Population standard deviation for alpha
  real<lower=0> tau_beta;  // Population standard deviation for beta

  // --- Non-Centered Subject-level parameters ---
  // We model the *standardized* offsets from the mean
  vector[J] alpha_raw; // Standardized subject-level effect (NCP)
  vector[J] beta_raw;  // Standardized subject-level effect (NCP)

  // --- Observation error ---
  real<lower=0> sigma; // Standard deviation of the observation noise
}

transformed parameters {
  // --- Re-center the parameters (NCP) ---
  // This is where we derive the individual subject parameters.
  // This parameterization is much more efficient for the sampler.
  vector<lower=0>[J] alpha = mu_alpha + alpha_raw * tau_alpha;
  vector<lower=0>[J] beta = mu_beta + beta_raw * tau_beta;

  // --- Calculate predicted y values (FULLY VECTORIZED) ---
  // No loops or if/else statements needed.
  // Stan functions like pow() are element-wise.
  // We index with N_subj_pos/neg to get the correct alpha/beta for each obs.

  vector[N] y_pred_pos = pow(x_pos, alpha[N_subj]);
  vector[N] y_pred_neg = -pow(x_neg, beta[N_subj]);
}

model {
  // --- Priors for Hyperparameters (Population Level) ---
  mu_alpha ~ normal(0.7, 10);
  mu_beta ~ normal(0.7, 10);
  tau_alpha ~ cauchy(0, 0.5);
  tau_beta ~ cauchy(0, 0.5);
  sigma ~ cauchy(0, 1.0);

  // --- Priors for Standardized Subject Effects (NCP) ---
  // These are the standardized offsets, so they are N(0, 1)
  alpha_raw ~ std_normal(); // Equivalent to normal(0, 1)
  beta_raw ~ std_normal();

  // --- Likelihood (FULLY VECTORIZED) ---
  // The two core likelihood statements, now much faster.
  y_pos ~ normal(y_pred_pos, sigma);
  y_neg ~ normal(y_pred_neg, sigma);
}

generated quantities {
  // We can still reconstruct the individual alphas and betas
  // for post-processing, even though they are defined in
  // transformed parameters.
  vector[J] individual_alpha = alpha;
  vector[J] individual_beta = beta;
}
