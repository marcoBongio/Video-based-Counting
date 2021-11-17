# CAN parameters
WIDTH = 320
HEIGHT = 320
PATCH_SIZE_PF = 8
#BE_CHANNELS = 1024

# TimeSformer parameters
WIDTH_TS = WIDTH // PATCH_SIZE_PF
HEIGHT_TS = HEIGHT // PATCH_SIZE_PF
IN_CHANS = 512
NUM_FRAMES = 4
PATCH_SIZE_TS = 2
DEPTH_TS = 2
NUM_HEADS = 8
EMBED_DIM = IN_CHANS * (PATCH_SIZE_TS * PATCH_SIZE_TS)
DIM_HEAD = EMBED_DIM // NUM_HEADS

MODEL_NAME = "320_1000_4f"
