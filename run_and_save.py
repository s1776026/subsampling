from datetime import datetime
import numpy as np
import pathlib
import pickle
import seaborn as sns

from simulate import ParticleSystem

sns.set()
sns.color_palette("colorblind")

file_path = "Test_Data/"
pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

parameters = {
    "particles": 100,
    "D": 1.0,
    "interaction_function": "ArcTan",
    "initial_dist_x": "pos_normal_dn",
    "T_end": 50,
    "dt": 0.01,
    "subsample": False,
}


def run_and_save(
    parameter="particles", values=[10], _filename="test", _filepath="Test_Data/"
):
    filename = str(_filename)
    filepath = str(_filepath)
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    kwargs = dict(parameters)
    with open(filepath + "parameters.txt", "w") as parameter_file:
        print(kwargs, file=parameter_file)

    print("\n Using default parameters:\n")
    for parameter_name, parameter_value in kwargs.items():
        print("\t{}:  {}".format(parameter_name, parameter_value))

    for value in values:
        print("\nSetting {} = {}".format(str(parameter), value))
        kwargs[parameter] = value
        startTime = datetime.now()
        np.random.seed(100)
        PS = ParticleSystem(**kwargs)
        t, x = PS.get_trajectories()
        test_data = {
            "Time": t,
            "Position": x,
        }
        print("Time to solve was  {} seconds".format(datetime.now() - startTime))

        file_name = filename + "{}".format(value)
        file_name = file_name.replace(".", "")
        pickle.dump(test_data, open(filepath + file_name, "wb"))
        print("Saved at {}\n".format(filepath + file_name))

        kwargs_sub = dict(kwargs)
        kwargs_sub["subsample"] = 10
        np.random.seed(100)
        PS_sub = ParticleSystem(**kwargs_sub)
        t, x = PS_sub.get_trajectories()
        test_data = {
            "Time": t,
            "Position": x,
        }
        print("Time to solve was  {} seconds".format(datetime.now() - startTime))

        file_name = filename + "{}".format(value)
        file_name = file_name.replace(".", "")
        pickle.dump(test_data, open(filepath + file_name + "sub", "wb"))
        print("Saved at {}\n".format(filepath + file_name + "sub"))


if __name__ == "__main__":
    values = np.floor(np.logspace(1, 3, 15))
    values = [int(value) for value in values]
    run_and_save(parameter="particles", values=values)
