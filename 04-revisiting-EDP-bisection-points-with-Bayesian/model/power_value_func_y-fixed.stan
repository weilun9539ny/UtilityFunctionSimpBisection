// Stan Model for Power Function (e.g., Value Function)
// FITS THE INVERSE MODEL: x = f(y)
// This correctly models the case where 'y' is fixed (predictor)
// and 'x' has the observation error (outcome).
//
// Based on the user's model structure:
//
// Positive: y = x^alpha  =>  Inverse: x = y^(1/alpha)
//   - We will estimate: beta_pos = 1/alpha
//
// Negative: y = -(-x)^beta => Inverse: x = -(-y)^(1/beta)
//   - We will estimate: beta_neg = 1/beta
//
data {
  // --- General ---
  int<lower=1> J; // Number of subjects
  int<lower=1> N; // Number of total observations
  array[N] int<lower=1, upper=J> N_subj; // Subject index

  // --- Data ---
  // 'y' is the predictor (fixed), 'x' is the outcome (has error)
  vector[N] y_pos; // Fixed y-values (y >= 0)
  vector[N] x_pos; // Observed x-values (x >= 0)

  vector[N] y_neg; // Fixed y-values (y > 0)
  vector[N] x_neg; // Observed x-values (x < 0)
}

parameters {
  // --- Population-level parameters (Hyperparameters) ---
  real<lower=0> mu_log_alpha_inverse;  // population mean for alpha inverse
  real<lower=0> mu_log_beta_inverse;  // population mean for beta inverse
  real<lower=0> tau_log_alpha_inverse;  // Population standard deviation for alpha inverse
  real<lower=0> tau_log_beta_inverse;  // Population standard deviation for beta inverse

  // --- Non-Centered Subject-level parameters ---
  vector[J] alpha_inverse_raw; // Standardized subject-level effect (NCP)
  vector[J] beta_inverse_raw; // Standardized subject-level effect (NCP)

  // --- Observation error ---
  // This is sigma on the 'x' variable
  real<lower=0> sigma;
}

transformed parameters {
  // --- Re-center the parameters (NCP) ---
  // Derive individual subject parameters
  vector<lower=0>[J] alpha_inverse = exp(mu_log_alpha_inverse + alpha_inverse_raw * tau_log_alpha_inverse);
  vector<lower=0>[J] beta_inverse = exp(mu_log_beta_inverse + beta_inverse_raw * tau_log_beta_inverse);

  // --- Calculate predicted x values (FULLY VECTORIZED) ---
  // We predict 'x' using 'y'
  // We must use the absolute value of y_neg for pow()
  vector[N] x_pred_pos = pow(y_pos, alpha_inverse[N_subj]);
  vector[N] x_pred_neg = pow(y_neg, beta_inverse[N_subj]);
}

model {
  // --- Priors for Hyperparameters (Population Level) ---
  // These priors are for beta_pos (1/alpha) and beta_neg (1/beta).
  // If you expect alpha/beta ~ 0.7, then 1/alpha ~ 1.4.
  // We'll set a generic prior around 1.0
  mu_log_alpha_inverse ~ normal(1.0, 1.0); // T[0,] truncates at 0
  mu_log_beta_inverse ~ normal(1.0, 1.0);

  tau_log_alpha_inverse ~ cauchy(0, 0.5);
  tau_log_beta_inverse ~ cauchy(0, 0.5);
  sigma ~ cauchy(0, 1.0);

  // --- Priors for Standardized Subject Effects (NCP) ---
  alpha_inverse_raw ~ std_normal(); // Equivalent to normal(0, 1)
  beta_inverse_raw ~ std_normal();

  // --- Likelihood (FULLY VECTORIZED) ---
  // The model predicts the observed 'x' values using the 'y' values.
  // This matches your loss function sum(x_obs - x_pred)^2
  x_pos ~ normal(x_pred_pos, sigma);
  x_neg ~ normal(x_pred_neg, sigma);
}

generated quantities {
  // Store the individual subject parameters for easy access.
  // beta_pos is the estimate for 1/alpha
  // beta_neg is the estimate for 1/beta
  vector[J] individual_alpha_inverse = alpha_inverse;
  vector[J] individual_beta_inverse = beta_inverse;

  // We can also calculate the 'original' alpha and beta
  // for easier interpretation, if desired.
  vector[J] alpha = 1.0 ./ alpha_inverse;
  vector[J] beta = 1.0 ./ beta_inverse;
}