"""
Test domain discovery functionality for evaluation pipeline.

Tests the domain extraction and discovery functions that map folder structures
to weather domains for FID computation.
"""
import pytest
from pathlib import Path
import tempfile
import csv

# Import the functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_generation import (
    extract_target_domain,
    discover_domains,
    discover_domains_from_manifest,
)


class TestExtractTargetDomain:
    """Tests for extract_target_domain function."""

    def test_extract_with_2_delimiter(self):
        """Test extraction with '2' delimiter (e.g., sunny_day2cloudy)."""
        assert extract_target_domain("sunny_day2cloudy") == "cloudy"
        assert extract_target_domain("clear2foggy") == "foggy"
        assert extract_target_domain("day2night") == "night"

    def test_extract_with_to_delimiter(self):
        """Test extraction with '_to_' delimiter (e.g., clear_day_to_cloudy)."""
        assert extract_target_domain("clear_day_to_cloudy") == "cloudy"
        assert extract_target_domain("sunny_to_rainy") == "rainy"
        assert extract_target_domain("day_to_night") == "night"
        assert extract_target_domain("clear_day_to_dawn_dusk") == "dawn_dusk"

    def test_extract_no_delimiter(self):
        """Test extraction with no delimiter (already a target domain)."""
        assert extract_target_domain("cloudy") == "cloudy"
        assert extract_target_domain("foggy") == "foggy"
        assert extract_target_domain("night") == "night"
        assert extract_target_domain("rainy") == "rainy"

    def test_extract_dataset_names(self):
        """Test extraction with dataset names (should return as-is)."""
        assert extract_target_domain("ACDC") == "ACDC"
        assert extract_target_domain("BDD100k") == "BDD100k"
        assert extract_target_domain("MapillaryVistas") == "MapillaryVistas"

    def test_to_delimiter_priority(self):
        """Test that '_to_' delimiter takes priority over '2'."""
        # If both delimiters exist, '_to_' should be used first
        assert extract_target_domain("test2something_to_cloudy") == "cloudy"

    def test_edge_cases(self):
        """Test edge cases.
        
        Note: When there's nothing after the delimiter, the function returns
        the original string (no extraction). This is acceptable behavior since
        real domain names follow specific patterns.
        """
        # Empty string
        assert extract_target_domain("") == ""
        # Only delimiter - returns as-is since no content after delimiter
        assert extract_target_domain("2") == "2"
        assert extract_target_domain("_to_") == "_to_"
        # Delimiter at end - returns original (no content to extract)
        assert extract_target_domain("sunny2") == "sunny2"
        assert extract_target_domain("sunny_to_") == "sunny_to_"


class TestDiscoverDomains:
    """Tests for discover_domains function (folder-based discovery)."""

    def test_discover_subfolders(self, tmp_path):
        """Test discovering domain subfolders with images."""
        # Create domain subfolders with images
        for domain in ["cloudy", "foggy", "night"]:
            domain_dir = tmp_path / domain
            domain_dir.mkdir()
            (domain_dir / "image1.png").touch()
        
        # Create empty subfolder (should be ignored)
        (tmp_path / "empty").mkdir()
        
        domains = discover_domains(tmp_path)
        assert set(domains) == {"cloudy", "foggy", "night"}

    def test_discover_root_with_images(self, tmp_path):
        """Test discovering root directory with images directly."""
        (tmp_path / "image1.png").touch()
        (tmp_path / "image2.jpg").touch()
        
        domains = discover_domains(tmp_path)
        assert domains == ["_root"]

    def test_discover_empty_directory(self, tmp_path):
        """Test discovering empty directory."""
        domains = discover_domains(tmp_path)
        assert domains == []


class TestDiscoverDomainsFromManifest:
    """Tests for discover_domains_from_manifest function."""

    def test_discover_from_target_domain_column(self, tmp_path):
        """Test discovering domains from manifest with target_domain column."""
        manifest_path = tmp_path / "manifest.csv"
        
        # Create manifest with target_domain column
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['gen_path', 'original_path', 'name', 'domain', 'dataset', 'target_domain'])
            writer.writerow(['/path/gen1.png', '/path/orig1.png', 'img1', 'cloudy', 'ACDC', 'cloudy'])
            writer.writerow(['/path/gen2.png', '/path/orig2.png', 'img2', 'foggy', 'BDD100k', 'foggy'])
            writer.writerow(['/path/gen3.png', '/path/orig3.png', 'img3', 'cloudy', 'MapillaryVistas', 'cloudy'])
            writer.writerow(['/path/gen4.png', '/path/orig4.png', 'img4', 'night', 'ACDC', 'night'])
        
        domains = discover_domains_from_manifest(manifest_path)
        assert set(domains) == {"cloudy", "foggy", "night"}

    def test_discover_from_domain_column_fallback(self, tmp_path):
        """Test discovering domains from manifest using domain column as fallback."""
        manifest_path = tmp_path / "manifest.csv"
        
        # Create manifest with only domain column (no target_domain)
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['gen_path', 'original_path', 'name', 'domain', 'dataset'])
            writer.writerow(['/path/gen1.png', '/path/orig1.png', 'img1', 'cloudy', 'ACDC'])
            writer.writerow(['/path/gen2.png', '/path/orig2.png', 'img2', 'rainy', 'BDD100k'])
        
        domains = discover_domains_from_manifest(manifest_path)
        assert set(domains) == {"cloudy", "rainy"}

    def test_discover_nonexistent_manifest(self, tmp_path):
        """Test discovering from nonexistent manifest file."""
        manifest_path = tmp_path / "nonexistent.csv"
        
        domains = discover_domains_from_manifest(manifest_path)
        assert domains == []

    def test_discover_manifest_no_domain_columns(self, tmp_path):
        """Test discovering from manifest without domain columns."""
        manifest_path = tmp_path / "manifest.csv"
        
        # Create manifest without domain columns
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['gen_path', 'original_path', 'name'])
            writer.writerow(['/path/gen1.png', '/path/orig1.png', 'img1'])
        
        domains = discover_domains_from_manifest(manifest_path)
        assert domains == []

    def test_discover_sorted_output(self, tmp_path):
        """Test that discovered domains are sorted."""
        manifest_path = tmp_path / "manifest.csv"
        
        # Create manifest with domains in random order
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['gen_path', 'original_path', 'domain'])
            writer.writerow(['/path/gen1.png', '/path/orig1.png', 'snowy'])
            writer.writerow(['/path/gen2.png', '/path/orig2.png', 'cloudy'])
            writer.writerow(['/path/gen3.png', '/path/orig3.png', 'night'])
            writer.writerow(['/path/gen4.png', '/path/orig4.png', 'foggy'])
        
        domains = discover_domains_from_manifest(manifest_path)
        assert domains == ["cloudy", "foggy", "night", "snowy"]


class TestDomainExtractionIntegration:
    """Integration tests for domain extraction scenarios from the issue."""

    def test_sustechgan_folder_pattern(self):
        """Test SUSTechGAN folder naming pattern: clear_day_to_cloudy."""
        folder_names = [
            "clear_day_to_cloudy",
            "clear_day_to_dawn_dusk",
            "clear_day_to_foggy",
            "clear_day_to_night",
            "clear_day_to_rainy",
            "clear_day_to_snowy",
        ]
        expected_domains = ["cloudy", "dawn_dusk", "foggy", "night", "rainy", "snowy"]
        
        extracted = [extract_target_domain(name) for name in folder_names]
        assert extracted == expected_domains

    def test_cyclegan_folder_pattern(self):
        """Test CycleGAN-style folder naming: sunny2cloudy."""
        folder_names = [
            "sunny2cloudy",
            "day2night",
            "clear2foggy",
        ]
        expected_domains = ["cloudy", "night", "foggy"]
        
        extracted = [extract_target_domain(name) for name in folder_names]
        assert extracted == expected_domains

    def test_weather_domains_passthrough(self):
        """Test that weather domain names pass through unchanged."""
        weather_domains = ["cloudy", "dawn_dusk", "foggy", "night", "rainy", "snowy"]
        
        for domain in weather_domains:
            assert extract_target_domain(domain) == domain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
