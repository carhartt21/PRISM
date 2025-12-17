# Manifest Generation Script Unification

## Summary

Unified `prepare_evaluation_manifest.py` and `generate_all_manifests.py` into a single script: `helper/generate_manifest.py`

## Changes Made

### New Script: `helper/generate_manifest.py`

**Features:**
- ✅ Single method mode: `--generated <path>`
- ✅ All methods mode: `--all`
- ✅ Missing only mode: `--all-missing`
- ✅ Handles all directory structures (domain/dataset, dataset/domain, flat)
- ✅ Domain normalization and task type detection
- ✅ Dataset identification from filename patterns
- ✅ Filename normalization (removes suffixes and prefixes)
- ✅ Weather domain indices for restoration task matching

**Usage Examples:**
```bash
# Single method
python helper/generate_manifest.py --generated /path/to/method --original /path/to/originals -o ./manifests/method

# All methods
python helper/generate_manifest.py --all

# Only missing
python helper/generate_manifest.py --all-missing

# Specific methods
python helper/generate_manifest.py --all --methods CUT cycleGAN
```

### Updated Scripts

1. **`scripts/run_evaluation`**
   - Changed from: `python3 helper/prepare_evaluation_manifest.py -o "$MANIFEST_FILE"`
   - Changed to: `python3 helper/generate_manifest.py -o "$(dirname "$MANIFEST_FILE")"`
   - Note: New script writes to directory, not directly to CSV file

2. **`helper/README.md`**
   - Updated documentation to reflect unified script
   - Added mode descriptions (--all, --all-missing)
   - Added migration note for deprecated scripts

3. **`README.md`**
   - Updated manifest generation examples
   - Added --all-missing mode example
   - Added single method example

### Deprecated Scripts

Moved to `helper/.deprecated/`:
- `prepare_evaluation_manifest.py`
- `generate_all_manifests.py`

Created migration guide in `helper/.deprecated/README.md`

## Key Differences from Old Scripts

### Output Location
- **Old (`prepare_evaluation_manifest.py`)**: `-o manifest.csv` (direct CSV path)
- **New**: `-o ./manifests/method` (directory containing manifest.csv and manifest.json)

### Modes
- **Old**: Separate scripts for single vs batch processing
- **New**: Unified with `--all` and `--all-missing` flags

### CSV Output Fields
Standardized to match evaluation script expectations:
- `gen_path`, `original_path`, `name`, `domain`, `dataset`, `target_domain`

## Testing

Test the script with:
```bash
# Help
python helper/generate_manifest.py --help

# Dry run
python helper/generate_manifest.py --all --dry-run -v

# Single method test
python helper/generate_manifest.py --generated /path/to/method --original /path/to/originals -o ./test_manifest
```

## Benefits

1. **Reduced code duplication** - ~1500 lines of duplicated logic eliminated
2. **Consistent behavior** - Same normalization and matching logic everywhere
3. **Easier maintenance** - Single codebase to update
4. **More flexible** - Three modes (single, all, all-missing) in one tool
5. **Better for automation** - `--all-missing` enables incremental processing
