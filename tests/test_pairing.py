"""
Test image pairing functionality
"""
import pytest
from pathlib import Path
import tempfile
import torch
from PIL import Image
import numpy as np

from utils.image_io import load_and_pair_images, find_image_files, match_by_filename


def create_dummy_image(path: Path, size: tuple = (64, 64)):
    """Create a dummy RGB image for testing."""
    # Create random RGB image
    img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.save(path)


class TestImagePairing:

    def test_find_image_files(self, tmp_path):
        """Test finding image files in directory."""
        # Create test images
        (tmp_path / "image1.png").touch()
        (tmp_path / "image2.jpg").touch()
        (tmp_path / "image3.JPEG").touch()
        (tmp_path / "not_image.txt").touch()

        files = find_image_files(tmp_path)
        assert len(files) == 3
        extensions = {f.suffix.lower() for f in files}
        assert extensions.issubset({'.png', '.jpg', '.jpeg'})

    def test_filename_matching(self, tmp_path):
        """Test matching by filename."""
        # Create directories
        gen_dir = tmp_path / "gen"
        real_dir = tmp_path / "real"
        gen_dir.mkdir()
        real_dir.mkdir()

        # Create test images with dummy content
        for i in range(3):
            create_dummy_image(gen_dir / f"image_{i}.png")
            create_dummy_image(real_dir / f"image_{i}.jpg")  # Different extension

        # Create unmatched image
        create_dummy_image(gen_dir / "unmatched.png")

        gen_files = find_image_files(gen_dir)
        real_files = find_image_files(real_dir)

        pairs = match_by_filename(gen_files, real_files)
        assert len(pairs) == 3  # Should match 3 pairs, ignore unmatched

        # Check that names are correct
        names = {name for _, _, name in pairs}
        expected_names = {f"image_{i}" for i in range(3)}
        assert names == expected_names

    def test_load_and_pair_images(self, tmp_path):
        """Test full loading and pairing pipeline."""
        # Create directories
        gen_dir = tmp_path / "gen"
        real_dir = tmp_path / "real"
        gen_dir.mkdir()
        real_dir.mkdir()

        # Create test images
        for i in range(2):
            create_dummy_image(gen_dir / f"test_{i}.png")
            create_dummy_image(real_dir / f"test_{i}.jpg")

        pairs = load_and_pair_images(gen_dir, real_dir)

        assert len(pairs) == 2

        # Check tensor properties
        gen_tensor, real_tensor, name = pairs[0]
        assert isinstance(gen_tensor, torch.Tensor)
        assert isinstance(real_tensor, torch.Tensor)
        assert gen_tensor.shape == (3, 299, 299)  # Default size
        assert real_tensor.shape == (3, 299, 299)
        assert 0 <= gen_tensor.min() and gen_tensor.max() <= 1  # [0, 1] range
        assert 0 <= real_tensor.min() and real_tensor.max() <= 1

    def test_empty_directories(self, tmp_path):
        """Test handling of empty directories."""
        gen_dir = tmp_path / "gen"
        real_dir = tmp_path / "real"
        gen_dir.mkdir()
        real_dir.mkdir()

        with pytest.raises(ValueError, match="No images found"):
            load_and_pair_images(gen_dir, real_dir)

    def test_custom_image_size(self, tmp_path):
        """Test custom image sizing."""
        gen_dir = tmp_path / "gen"
        real_dir = tmp_path / "real"
        gen_dir.mkdir()
        real_dir.mkdir()

        create_dummy_image(gen_dir / "test.png")
        create_dummy_image(real_dir / "test.jpg")

        custom_size = (128, 128)
        pairs = load_and_pair_images(gen_dir, real_dir, image_size=custom_size)

        gen_tensor, real_tensor, _ = pairs[0]
        assert gen_tensor.shape == (3, 128, 128)
        assert real_tensor.shape == (3, 128, 128)


if __name__ == "__main__":
    pytest.main([__file__])
