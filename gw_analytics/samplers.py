import pymc as pm
import pytensor.tensor as pt
import numpy as np

class GWNUTSSampler:
    def build_and_sample_model(self, draws=1500, tune=1000, chains=2):
        with pm.Model() as model:
            
            # 1. Priors
            chirp_mass = pm.Uniform("chirp_mass", lower=10.0, upper=50.0)
            lum_dist = pm.Uniform("luminosity_distance", lower=100.0, upper=2000.0)
            tc = pm.Uniform("tc", lower=0.4, upper=0.6)
            
            # ---------------------------------------------------------
            # 2. Waveform Math Graph (Adapted for Real Chirp Dynamics)
            # ---------------------------------------------------------
            
            t_shifted = self.time_array - tc
            
            # FIX 1: Create a smooth one-sided mask. Real waves end at the merger.
            # We use an inverse logit (sigmoid) function to smoothly fade the wave to zero 
            # after t_c. This keeps the math differentiable so NUTS' gradients don't crash.
            is_before_merger = pm.math.invlogit(-50.0 * t_shifted) 
            
            # Use absolute time for the math, add a tiny buffer to prevent division by zero
            tau = pm.math.abs(t_shifted) + 0.001 
            
            # FIX 2: True chirps INCREASE frequency as time gets closer to zero.
            # We invert the tau relationship so the frequency spikes at the merger.
            frequency = 30.0 + 5.0 * (chirp_mass**2) * (1.0 / (tau**0.5))
            
            phase = 2.0 * np.pi * frequency * t_shifted
            
            # FIX 3: Increase the baseline amplitude so it doesn't hit the 100 Mpc wall
            # Changed from 5e-19 to 5e-18 to accommodate loud signals like GW150914
            amplitude = 5e-18 / lum_dist
            
            # Apply the mask so the wave mathematically cuts off after the merger
            raw_expected_strain = amplitude * pm.math.sin(phase) * is_before_merger
            
            # ---------------------------------------------------------
            # 3. Scale and Link to Likelihood
            # ---------------------------------------------------------
            
            # Multiply by 1e21 to match the pre-scaled observed data array!
            scaled_template = raw_expected_strain * self.scale_factor
            
            # Likelihood Node linking to your scaled data and fixed sigma (1.0)
            pm.Normal(
                "likelihood", 
                mu=scaled_template, 
                sigma=self.noise_sigma, 
                observed=self.observed_strain
            )
            
            # 4. Run NUTS
            self.idata = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                init="jitter+adapt_diag",
                target_accept=0.90,
                progressbar=True
            )
        return self.idata