import numpy as np
import pymc as pm

class GWNUTSSampler:
    def __init__(self, time_array, observed_strain, noise_sigma):
        """
        NUTS Parameter Estimation suite aligned with actual GWOSC summary columns.
        Optimizes for Chirp Mass and Luminosity Distance.
        """
        self.t = np.array(time_array, dtype=np.float64)
        self.data = np.array(observed_strain, dtype=np.float64)
        self.sigma = float(noise_sigma)
        self.idata = None

    def build_and_sample_model(self, draws=2000, tune=1000, chains=2):
        """
        Compiles and samples using parameters present in the GWOSC EventTable.
        """
        print("Compiling real-parameter GR graph into PyMC...")
        
        with pm.Model() as self.model:
            # 1. Priors matching your column boundaries
            # Chirp Mass in Solar Masses
            chirp_mass = pm.Uniform("chirp_mass", lower=10.0, upper=50.0)
            
            # Luminosity Distance in Megaparsecs (Mpc)
            # Let's say our prior search window is between 100 Mpc and 2000 Mpc
            luminosity_distance = pm.Uniform("luminosity_distance", lower=100.0, upper=2000.0)

            # 2. Waveform Math Graph
            # Chirp mass governs how fast the frequency ramps up over time
            frequency = 30.0 + 5.0 * (chirp_mass**2) * (self.t**1.5)
            phase = 2.0 * np.pi * frequency * self.t
            
            # Amplitude scales inversely with Luminosity Distance (1 / d_L)
            # We use a baseline scaling constant (e.g., 5e-19) to match detector scales
            amplitude = 5e-19 / luminosity_distance
            expected_strain = amplitude * pm.math.sin(phase)

            # 3. Likelihood Node linking to your observed data array
            pm.Normal("likelihood", mu=expected_strain, sigma=self.sigma, observed=self.data)

            # 4. Execute NUTS
            print(f"Launching NUTS Sampler over {chains} parallel chains...")
            self.idata = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                init="jitter+adapt_diag",
                target_accept=0.90,
                return_inferencedata=True
            )
            
        print("Sampling complete.")
        return self.idata