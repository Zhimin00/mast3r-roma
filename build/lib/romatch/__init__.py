import os
from .models import roma_outdoor, tiny_roma_v1_outdoor, roma_indoor, romaSD_outdoor, romaSD16_8_outdoor, romaSD8_outdoor, romaDinoSD_outdoor, romaDinoSD_linear_outdoor, Mast3r_Roma_outdoor

DEBUG_MODE = False
RANK = int(os.environ.get('RANK', default = 0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1