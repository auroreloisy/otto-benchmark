# ____________ BASIC PARAMETERS _______________________________________________________________________________________
# Source-tracking POMDP
R_BAR = 2.5
# Neural network (NN) architecture
FC_LAYERS = 3  # number of hidden fully connected layers
FC_UNITS = 1024  # number of units, for each fully connected layers
# Learning rate for stochastic gradient descent
LEARNING_RATE = 5e-4  # we used manual annealing from 1e-3 to 1e-4
# Experience replay
MEMORY_SIZE = 1000  # number of transitions (s, s') to keep in memory
# Exploration: eps is the probability of taking a random action when executing the policy
E_GREEDY_DECAY = 20000   # timescale for eps decay, in number of training iterations
# Max number of training iterations
ALGO_MAX_IT = 10000000  # max number of training iterations
# Evaluation of the RL policy
EVALUATE_PERFORMANCE_EVERY = 5000  # how often is the RL policy evaluated, in number of training iterations
POLICY_REF = 1  # heuristic policy to use for comparison
N_RUNS_STATS = 500  # number of episodes used to compute the stats of a policy, set automatically if None
# Restart from saved model, if None start from scratch
MODEL_PATH = None  # path to saved model, e.g., "./models/20220201-230054/20220201-230054_model"
# Parallelization: how many episodes are computed in parallel (how many cores are used)
N_PARALLEL = 1    # -1 for using all cores, 1 for sequential (useful as parallel code may hang with larger NN)



