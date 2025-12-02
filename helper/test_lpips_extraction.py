
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import sys

def test_extraction():
    print("Initializing LPIPS...")
    try:
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    except Exception as e:
        print(f"Failed to init LPIPS: {e}")
        return

    vgg_net = lpips_model.net
    print(f"vgg_net type: {type(vgg_net)}")
    
    # Check for .net attribute
    if hasattr(vgg_net, 'net'):
        print("vgg_net has .net attribute")
        extractor = vgg_net.net
    else:
        print("vgg_net does NOT have .net attribute")
        extractor = vgg_net

    # Create dummy input
    dummy_input = torch.randn(2, 3, 64, 64) # Batch of 2
    # Normalize to [-1, 1]
    dummy_input = 2.0 * dummy_input - 1.0
    
    print("Attempting feature extraction...")
    try:
        features = extractor(dummy_input)
        print("Extraction successful!")
        print(f"Output type: {type(features)}")
        if isinstance(features, (list, tuple)):
            print(f"Number of feature maps: {len(features)}")
            for i, f in enumerate(features):
                print(f"Feature {i} shape: {f.shape}")
        else:
            print(f"Output shape: {features.shape}")
            
    except Exception as e:
        print(f"Extraction failed: {e}")
        # Print the error that would have happened with the original code
        try:
            vgg_net(dummy_input)
        except Exception as orig_e:
            print(f"Original code error would be: {orig_e}")

if __name__ == "__main__":
    test_extraction()
