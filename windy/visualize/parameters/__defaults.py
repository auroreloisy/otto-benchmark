# ____________ BASIC PARAMETERS _______________________________________________________________________________________
# Source-tracking POMDP
R_BAR = 2.5
# Policy
POLICY = 0  # -1=RL, O=infotaxis, 1=space-aware infotaxis, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
MODEL_PATH = None  # saved model for POLICY=-1, e.g., "../learn/models/20220201-230054/20220201-230054_model"
ALPHAVEC_PATH = None  # saved Perseus policy for POLICY=-2, e.g., "../../perseus/vf_rate_5.0_gamma_0.98_ic2_it_21_shaping_factor_0.1_shaping_power_1.0_nb_45000_epsilon_0.0_v2.pkl"
STEPS_AHEAD = 1  # number of anticipated moves, can be > 1 only for POLICY=0
# Setup
DRAW_SOURCE = True  # if False, episodes will continue until the source is almost surely found (Bayesian setting)
ZERO_HIT = False  # whether to enforce a series of zero hits
# Visualization
VISU_MODE = 2  # 0: run without video, 1: create video in the background, 2: create video and show live preview (slower)
FRAME_RATE = 5  # number of frames per second in the video
KEEP_FRAMES = False  # whether individual frames should be saved (otherwise frames are deleted, only the video is kept)
# ____________ ADVANCED PARAMETERS ____________________________________________________________________________________
# Stopping criteria
STOP_p = 1E-6  # episode stops when the probability that the source has been found is greater than 1 - STOP_p
STOP_t = 1000000  # maximum number of steps per episode
# Saving
RUN_NAME = None  # prefix used for all output files, if None will use timestamp



