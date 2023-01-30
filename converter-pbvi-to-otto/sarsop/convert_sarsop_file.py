"""Load Sarsop policy, extract useful data and save it."""
import numpy as np
import pickle

FILENAME = "sarsop_policy_windy_unshaped.pkl"
INPUT_FILENAME = "/home/aurore/Downloads/" + FILENAME
OUTPUT_FILENAME = "./" + FILENAME
OUTPUT = {
    "solver": "Sarsop",
    "alphas": None,
    "actions": None,
    "discount": "?",
    "shaping": "0",
    "shaping_coef": '',
}


def convert(input_filename, output):
    with open(input_filename, 'rb') as f:
        alphavecs = pickle.load(f)  # list of alpha_vec
    output["alphas"] = []
    output["actions"] = []
    for alphavec in alphavecs:
        output["alphas"].append(alphavec.data)
        if np.array_equal(alphavec.action, [1, 0]):
            action = 1
        elif np.array_equal(alphavec.action, [-1, 0]):
            action = 0
        elif np.array_equal(alphavec.action, [0, 1]):
            action = 3
        elif np.array_equal(alphavec.action, [0, -1]):
            action = 2
        else:
            raise Exception("This action is not defined")
        output["actions"].append(action)
    return output


if __name__ == "__main__":
    OUTPUT = convert(INPUT_FILENAME, OUTPUT)
    with open(OUTPUT_FILENAME, 'wb') as f:
        pickle.dump(OUTPUT, f, pickle.HIGHEST_PROTOCOL)
