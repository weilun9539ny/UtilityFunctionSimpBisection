// Hierarchical Bayesian CPT model in Stan
// Converted from Nilsson et al. (2011) WinBUGS implementation

// ---------- Stan code ----------

data {
  int<lower=1> N;              // number of observations (trials)
  int<lower=1> S;              // number of subjects

  int<lower=1> T1[S];             // starting trial for each subject
  int<lower=1> T2[S];             // ending trial for each subject

  // gamble attributes: allow 2-outcome gambles for A and B
  vector[N] xA1; vector[N] xA2; // outcomes and prob of outcome 1 for option A
  vector[N] xB1; vector[N] xB2; // outcomes and prob of outcome 1 for option B
  int<lower=0,upper=1> y[N];   // choice: 1 = chose option A, 0 = option B
}

parameters {
  // subject-level parameters (on transformed scales)
  vector[S] alpha_phi;   // probit-transformed α (0–1)
  vector[S] beta_phi;    // probit-transformed β (0–1)

  vector[S] log_lambda;  // log λ (loss aversion)
  vector[S] log_phi;     // log φ (sensitivity)

  // group-level means and sds
  real mu_alpha; real<lower=0> sigma_alpha;
  real mu_beta;  real<lower=0> sigma_beta;

  real mu_loglambda; real<lower=0> sigma_loglambda;
  real mu_logphi;    real<lower=0> sigma_logphi;
}

transformed parameters {
  // transform to original scales
  vector<lower=0,upper=1>[S] alpha;
  vector<lower=0,upper=1>[S] beta;
  vector<lower=0>[S] lambda;
  vector<lower=0>[S] phi;

  for (s in 1:S) {
    alpha[s]  = Phi(alpha_phi[s]);
    beta[s]   = Phi(beta_phi[s]);
    lambda[s] = exp(log_lambda[s]);
    phi[s]    = exp(log_phi[s]);
  }
}

model {
  // ----- Hyperpriors -----
  mu_alpha ~ normal(0,1);
  mu_beta  ~ normal(0,1);

  sigma_alpha ~ uniform(0,10);
  sigma_beta  ~ uniform(0,10);

  mu_loglambda ~ uniform(-2.30,1.61);   // corresponds to λ ~ [0.1,5]
  mu_logphi    ~ uniform(-2.30,1.61);   // corresponds to φ ~ [0.1,5]
  sigma_loglambda ~ uniform(0,1.13);
  sigma_logphi    ~ uniform(0,1.13);

  // ----- Subject-level priors -----
  alpha_phi ~ normal(mu_alpha, sigma_alpha);
  beta_phi  ~ normal(mu_beta, sigma_beta);

  log_lambda ~ normal(mu_loglambda, sigma_loglambda);
  log_phi    ~ normal(mu_logphi, sigma_logphi);

  // ----- Likelihood -----
  for (s in 1:S) {
    int t1 = T1[s]; int t2 = T2[s];
    for (n in t1:t2) {
      // subjective values for outcomes A
      real vA1 = (xA1[n] >= 0) ? pow(xA1[n], alpha[s]) : -lambda[s] * pow(-xA1[n], beta[s]);
      real vA2 = (xA2[n] >= 0) ? pow(xA2[n], alpha[s]) : -lambda[s] * pow(-xA2[n], beta[s]);
      real VA = 0.5 * (vA1 + vA2);

      // subjective values for outcomes B
      real vB1 = (xB1[n] >= 0) ? pow(xB1[n], alpha[s]) : -lambda[s] * pow(-xB1[n], beta[s]);
      real vB2 = (xB2[n] >= 0) ? pow(xB2[n], alpha[s]) : -lambda[s] * pow(-xB2[n], beta[s]);
      real VB = 0.5 * (vB1 + vB2);

      // choice likelihood
      y[n] ~ bernoulli_logit(phi[s] * (VA - VB));
    }
  }
}

generated quantities {
  vector[N] log_lik;
  for (s in 1:S) {
    int t1 = T1[s]; int t2 = T2[s];
    for (n in t1:t2) {
      real vA1 = (xA1[n] >= 0) ? pow(xA1[n], alpha[s]) : -lambda[s] * pow(-xA1[n], beta[s]);
      real vA2 = (xA2[n] >= 0) ? pow(xA2[n], alpha[s]) : -lambda[s] * pow(-xA2[n], beta[s]);
      real VA = 0.5 * (vA1 + vA2);

      real vB1 = (xB1[n] >= 0) ? pow(xB1[n], alpha[s]) : -lambda[s] * pow(-xB1[n], beta[s]);
      real vB2 = (xB2[n] >= 0) ? pow(xB2[n], alpha[s]) : -lambda[s] * pow(-xB2[n], beta[s]);
      real VB = 0.5 * (vB1 + vB2);

      log_lik[n] = bernoulli_logit_lpmf(y[n] | phi[s] * (VA - VB));
    }
  }
}
