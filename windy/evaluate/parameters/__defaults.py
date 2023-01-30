# ____________ BASIC PARAMETERS _______________________________________________________________________________________
# Source-tracking POMDP
R_BAR = 2.5
# Policy
POLICY = 0  # -1=RL, O=infotaxis, 1=space-aware infotaxis, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
MODEL_PATH = None  # saved model for POLICY=-1, e.g., "../learn/models/20220201-230054/20220201-230054_model"
ALPHAVEC_PATH = None  # saved Perseus policy for POLICY=-2, e.g., "../../perseus/vf_rate_5.0_gamma_0.98_ic2_it_21_shaping_factor_0.1_shaping_power_1.0_nb_45000_epsilon_0.0_v2.pkl"
STEPS_AHEAD = 1  # number of anticipated moves, can be > 1 only for POLICY=0
# Setup
DRAW_SOURCE = False  # if False, episodes will continue until the source is almost surely found (Bayesian setting)
# Parallelization
N_PARALLEL = -1  # -1 for using all cores, 1 for sequential (useful as parallel code may hang with larger NN)
# ____________ ADVANCED PARAMETERS ____________________________________________________________________________________
# Criteria for terminating an episode
STOP_t = None  # maximum number of steps per episode, set automatically if None
STOP_p = 1E-6  # episode stops when the probability that the source has been found is greater than 1 - STOP_p
# Statistics computation
ADAPTIVE_N_RUNS = True  # if true, N_RUNS is increased until the estimated error is less than REL_TOL
N_RUNS = None  # number of episodes to compute (starting guess if ADAPTIVE_N_RUNS)
REL_TOL = 0.01  # tolerance on the relative error on the mean number of steps to find the source (if ADAPTIVE_N_RUNS)
MAX_N_RUNS = 100000  # maximum number of runs (if ADAPTIVE_N_RUNS)
# Saving
RUN_NAME = None  # prefix used for all output files, if None will use timestamp



