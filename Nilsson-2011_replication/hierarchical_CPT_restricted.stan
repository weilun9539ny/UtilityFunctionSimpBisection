// Hierarchical Bayesian CPT model in Stan
// Converted from Nilsson et al. (2011) WinBUGS implementation

// ---------- Stan code ----------

data {
  int<lower=1> N;              // number of observations (trials)
  int<lower=1> S;              // number of subjects
  int<lower=1,upper=S> subj[N]; // subject id per trial
  int<lower=0,upper=1> y[N];   // choice: 1 = chose option A, 0 = option B

  // gamble attributes: allow 2-outcome gambles for A and B
  vector[N] xA1; vector[N] xA2; vector[N] pA; // outcomes and prob of outcome 1 for option A
  vector[N] xB1; vector[N] xB2; vector[N] pB; // outcomes and prob of outcome 1 for option B
}

parameters {
  // subject-level parameters (on transformed scales)
  vector[S] alpha_phi;   // probit-transformed α (0–1)
  vector[S] gamma_phi;   // probit-transformed γ (0–1)
  vector[S] delta_phi;   // probit-transformed δ (0–1)

  vector[S] log_lambda;  // log λ (loss aversion)
  vector[S] log_phi;     // log φ (sensitivity)

  // group-level means and sds
  real mu_alpha; real<lower=0> sigma_alpha;
  real mu_gamma; real<lower=0> sigma_gamma;
  real mu_delta; real<lower=0> sigma_delta;

  real mu_loglambda; real<lower=0> sigma_loglambda;
  real mu_logphi;    real<lower=0> sigma_logphi;
}

transformed parameters {
  // transform to original scales
  vector<lower=0,upper=1>[S] alpha;
  vector<lower=0,upper=1>[S] gamma;
  vector<lower=0,upper=1>[S] delta;
  vector<lower=0>[S] lambda;
  vector<lower=0>[S] phi;

  for (s in 1:S) {
    alpha[s]  = Phi(alpha_phi[s]);
    gamma[s]  = 0.4 + 0.6 * Phi(gamma_phi[s]);  // gamma in (0.4, 1)
    delta[s]  = 0.4 + 0.6 * Phi(delta_phi[s]);  // delta in (0.4, 1)
    lambda[s] = exp(log_lambda[s]);
    phi[s]    = exp(log_phi[s]);
  }
}

model {
  // ----- Hyperpriors -----
  mu_alpha ~ normal(0,1);
  mu_gamma ~ normal(0,1);
  mu_delta ~ normal(0,1);

  sigma_alpha ~ uniform(0,10);
  sigma_gamma ~ uniform(0,10);
  sigma_delta ~ uniform(0,10);

  mu_loglambda ~ uniform(-6.90, 2.3);   // corresponds to λ ~ [0.001, 10]
  mu_logphi    ~ uniform(-6.90, 2.3);   // corresponds to φ ~ [0.001, 10]
  sigma_loglambda ~ uniform(0,1.13);
  sigma_logphi    ~ uniform(0,1.13);

  // ----- Subject-level priors -----
  alpha_phi ~ normal(mu_alpha, sigma_alpha);
  gamma_phi ~ normal(mu_gamma, sigma_gamma);
  delta_phi ~ normal(mu_delta, sigma_delta);

  log_lambda ~ normal(mu_loglambda, sigma_loglambda);
  log_phi    ~ normal(mu_logphi, sigma_logphi);

  // ----- Likelihood -----
  for (n in 1:N) {
    int s = subj[n];  // index for subjects

    // subjective values for outcomes A
    real vA1 = (xA1[n] >= 0) ? pow(xA1[n], alpha[s]) : -lambda[s] * pow(-xA1[n], alpha[s]);
    real vA2 = (xA2[n] >= 0) ? pow(xA2[n], alpha[s]) : -lambda[s] * pow(-xA2[n], alpha[s]);

    // subjective values for outcomes B
    real vB1 = (xB1[n] >= 0) ? pow(xB1[n], alpha[s]) : -lambda[s] * pow(-xB1[n], alpha[s]);
    real vB2 = (xB2[n] >= 0) ? pow(xB2[n], alpha[s]) : -lambda[s] * pow(-xB2[n], alpha[s]);

    // probability weighting for lottery A (Tversky-Kahneman power form)
    real wA;
    real VA;
    if (xA1[n] >= 0 && xA2[n] >= 0) {
      // both gain: both use gamma
      wA = pow(pA[n], gamma[s]) / pow(pow(pA[n], gamma[s]) + pow(1-pA[n], gamma[s]), 1/gamma[s]);
      VA = wA * vA1 + (1-wA) * vA2;
    } else if (xA1[n] <= 0 && xA2[n] <= 0) {
      // both loss: both use delta
      wA = pow(pA[n], delta[s]) / pow(pow(pA[n], delta[s]) + pow(1-pA[n], delta[s]), 1/delta[s]);
      VA = wA * vA1 + (1-wA) * vA2;
    } else {
      // mixed: use gamma and delta
      if (xA1[n] >= 0) {
        real wg = pow(pA[n], gamma[s]) / pow(pow(pA[n],gamma[s]) + pow(1-pA[n],gamma[s]), 1/gamma[s]);
        real wl = pow(1-pA[n], delta[s]) / pow(pow(1-pA[n],delta[s]) + pow(pA[n],delta[s]), 1/delta[s]);
        VA = wg * vA1 + wl * vA2;
      } else {
        real wg = pow(1-pA[n], gamma[s]) / pow(pow(1-pA[n],gamma[s]) + pow(pA[n],gamma[s]), 1/gamma[s]);
        real wl = pow(pA[n], delta[s]) / pow(pow(pA[n],delta[s]) + pow(1-pA[n],delta[s]), 1/delta[s]);
        VA = wg * vA2 + wl * vA1;
      }
    }
    
    // probability weighting for lottery B (Tversky-Kahneman power form)
    real wB; 
    real VB;
    if (xB1[n] >= 0 && xB2[n] >= 0) {
      // both gain: both use gamma
      wB = pow(pB[n], gamma[s]) / pow(pow(pB[n], gamma[s]) + pow(1-pB[n], gamma[s]), 1/gamma[s]);
      VB = wB * vB1 + (1-wB) * vB2;
    } else if (xB1[n] <= 0 && xB2[n] <= 0) {
      // both loss: both use delta
      wB = pow(pB[n], delta[s]) / pow(pow(pB[n], delta[s]) + pow(1-pB[n], delta[s]), 1/delta[s]);
      VB = wB * vB1 + (1-wB) * vB2;
    } else {
      // mixed: use gamma and delta
      if (xB1[n] >= 0) {
        real wg = pow(pB[n], gamma[s]) / pow(pow(pB[n],gamma[s]) + pow(1-pB[n],gamma[s]), 1/gamma[s]);
        real wl = pow(1-pB[n], delta[s]) / pow(pow(1-pB[n],delta[s]) + pow(pB[n],delta[s]), 1/delta[s]);
        VB = wg * vB1 + wl * vB2;
      } else {
        real wg = pow(1-pB[n], gamma[s]) / pow(pow(1-pB[n],gamma[s]) + pow(pB[n],gamma[s]), 1/gamma[s]);
        real wl = pow(pB[n], delta[s]) / pow(pow(pB[n],delta[s]) + pow(1-pB[n],delta[s]), 1/delta[s]);
        VB = wg * vB2 + wl * vB1;
      }
    }

    // choice likelihood
    y[n] ~ bernoulli_logit(phi[s] * (VA - VB));
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    int s = subj[n];
    real vA1 = (xA1[n] >= 0) ? pow(xA1[n], alpha[s]) : -lambda[s] * pow(-xA1[n], alpha[s]);
    real vA2 = (xA2[n] >= 0) ? pow(xA2[n], alpha[s]) : -lambda[s] * pow(-xA2[n], alpha[s]);
    real vB1 = (xB1[n] >= 0) ? pow(xB1[n], alpha[s]) : -lambda[s] * pow(-xB1[n], alpha[s]);
    real vB2 = (xB2[n] >= 0) ? pow(xB2[n], alpha[s]) : -lambda[s] * pow(-xB2[n], alpha[s]);
    
    real VA; real VB;
    real wA; real wB;
    // probability weighting for lottery A (Tversky-Kahneman power form)
    if (xA1[n] >= 0 && xA2[n] >= 0) {
      // both gain: both use gamma
      wA = pow(pA[n], gamma[s]) / pow(pow(pA[n], gamma[s]) + pow(1-pA[n], gamma[s]), 1/gamma[s]);
      VA = wA * vA1 + (1-wA) * vA2;
    } else if (xA1[n] <= 0 && xA2[n] <= 0) {
      // both loss: both use delta
      wA = pow(pA[n], delta[s]) / pow(pow(pA[n], delta[s]) + pow(1-pA[n], delta[s]), 1/delta[s]);
      VA = wA * vA1 + (1-wA) * vA2;
    } else {
      // mixed: use gamma and delta
      if (xA1[n] >= 0) {
        real wg = pow(pA[n], gamma[s]) / pow(pow(pA[n],gamma[s]) + pow(1-pA[n],gamma[s]), 1/gamma[s]);
        real wl = pow(1-pA[n], delta[s]) / pow(pow(1-pA[n],delta[s]) + pow(pA[n],delta[s]), 1/delta[s]);
        VA = wg * vA1 + wl * vA2;
      } else {
        real wg = pow(1-pA[n], gamma[s]) / pow(pow(1-pA[n],gamma[s]) + pow(pA[n],gamma[s]), 1/gamma[s]);
        real wl = pow(pA[n], delta[s]) / pow(pow(pA[n],delta[s]) + pow(1-pA[n],delta[s]), 1/delta[s]);
        VA = wg * vA2 + wl * vA1;
      }
    }
    
    // probability weighting for lottery B (Tversky-Kahneman power form)
    if (xB1[n] >= 0 && xB2[n] >= 0) {
      // both gain: both use gamma
      wB = pow(pB[n], gamma[s]) / pow(pow(pB[n], gamma[s]) + pow(1-pB[n], gamma[s]), 1/gamma[s]);
      VB = wB * vB1 + (1-wB) * vB2;
    } else if (xB1[n] <= 0 && xB2[n] <= 0) {
      // both loss: both use delta
      wB = pow(pB[n], delta[s]) / pow(pow(pB[n], delta[s]) + pow(1-pB[n], delta[s]), 1/delta[s]);
      VB = wB * vB1 + (1-wB) * vB2;
    } else {
      // mixed: use gamma and delta
      if (xB1[n] >= 0) {
        real wg = pow(pB[n], gamma[s]) / pow(pow(pB[n],gamma[s]) + pow(1-pB[n],gamma[s]), 1/gamma[s]);
        real wl = pow(1-pB[n], delta[s]) / pow(pow(1-pB[n],delta[s]) + pow(pB[n],delta[s]), 1/delta[s]);
        VB = wg * vB1 + wl * vB2;
      } else {
        real wg = pow(1-pB[n], gamma[s]) / pow(pow(1-pB[n],gamma[s]) + pow(pB[n],gamma[s]), 1/gamma[s]);
        real wl = pow(pB[n], delta[s]) / pow(pow(pB[n],delta[s]) + pow(1-pB[n],delta[s]), 1/delta[s]);
        VB = wg * vB2 + wl * vB1;
      }
    }

    log_lik[n] = bernoulli_logit_lpmf(y[n] | phi[s] * (VA - VB));
  }
}

