import numpy as np
import matplotlib.pyplot as plt
import interactionfunctions as Bs


class ParticleSystem:
    def __init__(
        self,
        particles=10,
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
            "Uniform": Bs.uniform,
            "Zero": Bs.zero,
            "Neg Exp": Bs.neg_exp,
        }
        try:
            self.B = interaction_functions[self.interaction_function]
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
            self.x0 = ic_xs[self.initial_dist_x]
        except (KeyError, TypeError) as error:
            if isinstance(self.initial_dist_x, (list, tuple, np.ndarray)):
                print("Using ndarray for velocity distribution")
                self.x0 = self.initial_dist_x
            elif self.initial_dist_x is None:
                print("Using default, positive normal distribution\n")
                self.x0 = np.random.normal(
                    loc=1, scale=np.sqrt(self.D), size=self.particles
                )
            else:
                print(
                    "{} is not a valid keyword. Valid initial conditions for positions are {}".format(
                        error, list(ic_xs.keys())
                    )
                )

    def calculate_interaction(self, x_curr):
        """Calculate interaction term of the full particle system

            Args:
                x_curr: np.array of current particle positions
                B: interaction function
                L: domain length, float

            Returns:
                array: The calculated interaction at the current time step for each
                    particle

            See Also:
                :py:mod:`~particle.interactionfunctions`
        """
        interaction_vector = np.zeros(len(x_curr))
        for particle, position in enumerate(x_curr):
            particle_interaction = self.B(x_curr)
            weighted_avg = position * np.sum(particle_interaction)
            interaction_vector[particle] = weighted_avg / len(x_curr)
        return interaction_vector

    def EM_scheme_step(self):
        x = self.x0
        self.interaction_data = []
        while 1:
            yield x
            interaction = self.calculate_interaction(x)
            self.interaction_data.append(interaction)
            x = (
                x
                + interaction * self.dt
                + np.sqrt(2 * self.D * self.dt) * np.random.normal(size=self.particles)
            )

    def get_trajectories(self):
        """ Returns n_samples from a given algorithm. """
        self.set_inital_conditions()
        step = self.EM_scheme_step()
        t = np.arange(0, self.T_end + self.dt, self.dt)
        N = len(t) - 1
        x = [next(step) for _ in range(N + 1)]

        return t, np.array(x)


if __name__ == "__main__":
    parameters = {
        "particles": 100,
        "interaction_function": "Neg Exp",
        "initial_dist_x": np.zeros(100),
    }

    PS = ParticleSystem(**parameters)
    t, x = PS.get_trajectories()
    plt.plot(np.tile(t, (5,1)).T, x[:, :5],alpha=0.1)
    plt.plot(t, np.mean(x, axis=1),'r--')


    plt.show()
