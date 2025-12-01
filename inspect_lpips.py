
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def inspect_lpips():
    try:
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        print(f"lpips_model type: {type(lpips_model)}")
        print(f"lpips_model.net type: {type(lpips_model.net)}")
        
        # Try to inspect what .net is
        print("\nStructure of lpips_model.net:")
        print(lpips_model.net)
        
        # Create a dummy input
        dummy_input = torch.randn(1, 3, 64, 64)
        
        print("\nAttempting to call lpips_model.net(dummy_input)...")
        try:
            output = lpips_model.net(dummy_input)
            print("Success! Output type:", type(output))
        except Exception as e:
            print(f"Failed: {e}")
            
    except Exception as e:
        print(f"Error initializing LPIPS: {e}")

if __name__ == "__main__":
    inspect_lpips()
