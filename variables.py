WIDTH = 224
HEIGHT = 224
PATCH_SIZE_PF = 8
BE_CHANNELS = 1024

DIM_TS = WIDTH // PATCH_SIZE_PF
IN_CHANS = 512
NUM_FRAMES = 2
PATCH_SIZE_TS = 1
DEPTH_TS = 4
NUM_HEADS = 8
EMBED_DIM = IN_CHANS * (PATCH_SIZE_TS * PATCH_SIZE_TS)
DIM_HEAD = EMBED_DIM // NUM_HEADS
AVG_POOL_SIZE = 1024

MODEL_NAME = "weekend"
