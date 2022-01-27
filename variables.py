# CAN parameters
WIDTH = 640
HEIGHT = 360
PATCH_SIZE_PF = 8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODE = 'concat'
if MODE == 'concat':
    BE_CHANNELS = 1024
elif MODE == 'weighted':
    BE_CHANNELS = 512

MODEL_NAME = "timesformer_640x320_3frames_1layer_no-spatial-att"

# TimeSformer parameters
WIDTH_TS = WIDTH // PATCH_SIZE_PF
HEIGHT_TS = HEIGHT // PATCH_SIZE_PF
IN_CHANS = 512
NUM_FRAMES = 4
PATCH_SIZE_TS = 2
DEPTH_TS = 2
NUM_HEADS = 8
EMBED_DIM = 2048
DIM_HEAD = EMBED_DIM // NUM_HEADS
