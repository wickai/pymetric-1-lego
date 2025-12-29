
import sys
import os
sys.path.append(os.getcwd())

from metric.core.config import cfg
import metric.core.config as config
from metric.core.builders import build_model

def test_build():
    cfg_file = "configs/nas/MbV2Nas_lw_imagenet_4card_5blk.yaml"
    print(f"Loading config from {cfg_file}")
    cfg.merge_from_file(cfg_file)
    
    print("Building model...")
    try:
        model = build_model()
        print("Model built successfully!")
        print(model)
        
        # Check if it has the right number of blocks
        # The JSON has 6 op codes.
        # MobileNetV2 features should have some layers.
        # Check if we can access op_codes if possible, or just print structure.
        
    except Exception as e:
        print(f"Failed to build model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_build()
