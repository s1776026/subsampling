import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

sns.set()
file_path = "Test_Data/"
subdir = ""

print("Reading from", file_path + subdir)
# Finds all data in subfolder

mypath = file_path + subdir
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
onlyfiles = [file for file in onlyfiles if ".txt" not in file]
print("\nData Files:\n")
print(*onlyfiles, sep="\n")
subsample_files = [file for file in onlyfiles if "sub" in file]
full_files = [file for file in onlyfiles if file not in subsample_files]

difference = []
N = []
for i in range(len(full_files)):
    full_data = pickle.load(open(file_path + subdir + full_files[i], "rb"))
    sub_data = pickle.load(open(file_path + subdir + subsample_files[i], "rb"))

    t_full = full_data["Time"]
    x_full = full_data["Position"]

    t_sub = sub_data["Time"]
    x_sub = sub_data["Position"]

    difference.append(np.abs(np.mean(x_full[-1, ]) - np.mean(x_sub[-1, ])))
    N.append(len(x_full[0, ]))

print(difference)
print(np.sort(N))
plt.plot(np.sort(N), difference)
plt.show()
