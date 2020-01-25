from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import particle.plotting as myplot
import particle.interactionfunctions as phis


class ParticleSystem:
    def __init__(
        self,
        particles=100,
        D=1,
        initial_dist_x=None,
        interaction_function="Zero",
        dt=0.1,
        T_end=100,
        stopping_time=False,
    ):
        self.particles = particles
        self.D = D
        self.interaction_function = interaction_function
        self.dt = dt
        self.T_end = T_end
        self.initial_dist_x = initial_dist_x

        interaction_functions = {
            "Garnier": lambda x: phis.Garnier(x, self.L),
            "Uniform": phis.uniform,
            "Zero": phis.zero,
            "Indicator": lambda x: phis.indicator(x, self.L),
            "Smoothed Indicator": phis.smoothed_indicator,
            "Gamma": lambda x: phis.gamma(x, self.gamma, self.L),
        }
        try:
            self.phi = interaction_functions[interaction_function]
        except KeyError as error:
            print(
                "{} is not valid. Valid interactions are {}".format(
                    error, list(interaction_functions.keys())
                )
            )
            return
            
    def set_inital_conditions(self):
        # Initial condition in velocity
        ic_xs = {
            "pos_normal_dn": np.random.normal(
                loc=1, scale=np.sqrt(2), size=self.particles
            ),
            "neg_normal_dn": np.random.normal(
                loc=-1, scale=np.sqrt(2), size=self.particles
            ),
            "uniform_dn": np.random.uniform(low=0, high=1, size=self.particles),
            "cauchy_dn": np.random.standard_cauchy(size=self.particles),
            "gamma_dn": np.random.gamma(shape=7.5, scale=1.0, size=self.particles),
        }
        # Try using dictionary to get IC, if not check if input is array, else use a
        # default IC
        try:
            self.x0 = ic_vs[self.initial_dist_x]
        except (KeyError, TypeError) as error:
            if isinstance(self.initial_dist_x, (list, tuple, np.ndarray)):
                print("Using ndarray for velocity distribution")
                self.x0 = self.initial_dist_x
            elif self.initial_dist_x is None:
                print("Using default, positive normal distrbution\n")
                self.x0 = np.random.normal(
                    loc=1, scale=np.sqrt(self.D), size=self.particles
                )
            else:
                print(
                    "{} is not a valid keyword. Valid initial conditions for positions are {}".format(
                        error, list(ic_xs.keys())
                    )
                )
