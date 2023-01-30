# ____________ BASIC PARAMETERS _______________________________________________________________________________________
# Source-tracking POMDP
N_DIMS = 2  # number of space dimensions (1D, 2D, ...)
LAMBDA_OVER_DX = 3.0  # dimensionless problem size
R_DT = 2.0  # dimensionless source intensity
# Policy
POLICY = -1  # -2=PBVI, -1=DRL, O=infotaxis, 1=space-aware infotaxis, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
if POLICY == -1:
    # DRL policy
    # provided model
    MODEL_PATH = "../../zoo/isotropic-53x53_drl"
    # custom trained model
    # MODEL_PATH = "../learn/models/20220201-230054/20220201-230054_model"  # to be adapted to your model
elif POLICY == -2:
    # PBVI policy (Sarsop, Perseus)
    # provided Sarsop
    ALPHAVEC_PATH = "../../zoo/isotropic-53x53_sarsop.pkl"
    # provided Sarsop-Light
    # ALPHAVEC_PATH = "../../zoo/isotropic-53x53_sarsop-light.pkl"
    # provided Perseus
    # ALPHAVEC_PATH = "../../zoo/isotropic-53x53_perseus.pkl"
# Setup
DRAW_SOURCE = True  # if False, episodes will continue until the source is almost surely found (Bayesian setting)
ZERO_HIT = False  # whether to enforce a series of zero hits
# Visualization
VISU_MODE = 2  # 0: run without video, 1: create video in the background, 2: create video and show live preview (slower)
FRAME_RATE = 5  # number of frames per second in the video
KEEP_FRAMES = False  # whether individual frames should be saved (otherwise frames are deleted, only the video is kept)

