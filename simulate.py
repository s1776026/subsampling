from datetime import datetime as datetime
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
        subsample=False,
    ):
        self.particles = particles
        self.D = D
        self.interaction_function = interaction_function
        self.dt = dt
        self.T_end = T_end
        self.initial_dist_x = initial_dist_x
        self.subsample = subsample

        interaction_functions = {
            "Uniform": Bs.uniform,
            "Zero": Bs.zero,
            "Neg Exp": Bs.neg_exp,
            "ArcTan": Bs.arctan,
            "ExpProd": Bs.exp_product,
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

        if self.subsample:
            print("Subsampling with sample size {}".format(self.subsample))
            if self.subsample > self.particles:
                print(
                    "Subsample {} is greater than particle count {}".format(
                        self.subsample, self.particles
                    )
                )
                print("Setting subsample equal to particle count")
                self.subsample = self.particles

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
                msg = "{} is not a valid keyword. Valid initial conditions"
                +"for positions are {}"
                print(msg.format(error, list(ic_xs.keys())))

    def calculate_interaction(self, x_curr):
        """Calculate interaction term of the full particle system

            Args:
                x_curr: np.array of current particle positions
                B: interaction function

            Returns:
                array: The calculated interaction at the current time step for each
                    particle

            See Also:
                :py:mod:`~particle.interactionfunctions`
        """
        particle_interaction = np.zeros(len(x_curr))
        for particle, position in enumerate(x_curr):
            if self.subsample:
                subsample_mask = np.random.choice(len(x_curr), size=self.subsample)
                x_subsample = x_curr[subsample_mask]
                particle_interaction[particle] = (1/ len(x_subsample)) * np.sum(
                    self.B(x_subsample*position)
                )
            else:
                subsample_mask = np.random.choice(len(x_curr), size=5)
                particle_interaction[particle] = (1/ len(x_curr)) * np.sum(
                    self.B(x_curr*position)
                )
        return particle_interaction

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
        "D": 1.0,
        "interaction_function": "ExpProd",
        "initial_dist_x": 0.2 * np.ones(100),
        "T_end": 5,
        "dt": 0.01,
    }
    import seaborn as sns

    sns.set()
    np.random.seed(100)
    start = datetime.now()
    PS = ParticleSystem(**parameters)
    t, x = PS.get_trajectories()
    print("Time taken for full sim: {}".format(datetime.now() - start))

    subs_parameters = parameters
    subs_parameters["subsample"] = 5
    np.random.seed(100)
    start = datetime.now()
    PS_sub = ParticleSystem(**subs_parameters)
    t, x_sub = PS_sub.get_trajectories()
    # print(PS_sub.interaction_data)
    # print(PS.interaction_data)
    print("Time taken for subsample sim: {}".format(datetime.now() - start))
    with sns.color_palette("coolwarm", 200):
        plt.plot(np.tile(t, (100, 1)).T, x[:, :100], label="Full", alpha=0.8)
        plt.plot(np.tile(t, (100, 1)).T, x_sub[:, :100], label="Subsample", alpha=0.8)
    plt.legend()
    plt.plot(t, np.mean(x_sub, axis=1), "r--")
    plt.plot(t, np.mean(x, axis=1), "g--")
    plt.ylim((-100,100))
    plt.show()
