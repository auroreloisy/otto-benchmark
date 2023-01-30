# ____________ BASIC PARAMETERS _______________________________________________________________________________________
# Source-tracking POMDP
N_DIMS = 2  # number of dimension (1D, 2D, ...)
LAMBDA_OVER_DX = 1.0  # dimensionless problem size
R_DT = 1.0  # dimensionless source intensity
# Policy
POLICY = -1  # -2=PBVI, -1=DRL, O=infotaxis, 1=space-aware infotaxis, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
if POLICY == -1:
    # DRL policy
    # provided model
    MODEL_PATH = "../../zoo/isotropic-19x19_drl"
    # custom trained model
    # MODEL_PATH = "../learn/models/20220201-230054/20220201-230054_model"  # to be adapted to your model
elif POLICY == -2:
    # PBVI policy (Sarsop, Perseus)
    # provided Sarsop
    ALPHAVEC_PATH = "../../zoo/isotropic-19x19_sarsop.pkl"
    # provided Sarsop-Light
    # ALPHAVEC_PATH = "../../zoo/isotropic-19x19_sarsop-light.pkl"
    # provided Perseus
    # ALPHAVEC_PATH = "../../zoo/isotropic-19x19_perseus.pkl"
# Parallelization: -1 for using all cores, 1 for sequential (useful as parallel code may hang with larger NN)
if POLICY < 0:
    N_PARALLEL = 1
else:
    N_PARALLEL = -1
# ____________ ADVANCED PARAMETERS ____________________________________________________________________________________
# Statistics computation
ADAPTIVE_N_RUNS = False  # if true, N_RUNS is increased until the estimated error is less than REL_TOL
N_RUNS = 50000  # number of episodes to compute (starting guess if ADAPTIVE_N_RUNS)

