#!/bin/bash
"""
Count files per subfolder in a directory using shell commands.

Usage:
    ./count_files.sh --root /path/to/directory [--recursive] [--extensions jpg png] [--format csv|json|plain] [--output file]

Example:
    ./count_files.sh --root /scratch/aaa_exchange/AWARE/GENERATED_IMAGES/cycleGAN --recursive --format csv
"""

# Parse arguments
ROOT=""
RECURSIVE=false
EXTENSIONS=()
FORMAT="plain"
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --recursive)
      RECURSIVE=true
      shift
      ;;
    --extensions)
      shift
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        EXTENSIONS+=("$1")
        shift
      done
      ;;
    --format)
      FORMAT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --root <dir> [--recursive] [--extensions ext1 ext2] [--format plain|csv|json] [--output file]"
      exit 1
      ;;
  esac
done

if [ -z "$ROOT" ]; then
  echo "Error: --root is required"
  exit 1
fi

if [ ! -d "$ROOT" ]; then
  echo "Error: $ROOT is not a directory"
  exit 1
fi

# Function to build find name expression
build_names() {
  local names=""
  for ext in "${EXTENSIONS[@]}"; do
    if [ -z "$names" ]; then
      names="-name \"*.$ext\""
    else
      names="$names -o -name \"*.$ext\""
    fi
  done
  echo "$names"
}

# Declare associative array for counts
declare -A counts

if [ "$RECURSIVE" = true ]; then
  # Recursive: find all files, group by dirname
  if [ ${#EXTENSIONS[@]} -gt 0 ]; then
    names=$(build_names)
    find "$ROOT" -type f \( $names \) -printf '%h\n' | sort | uniq -c | awk '{print $2 "," $1}' | while IFS=',' read -r folder count; do
      counts["$folder"]=$count
    done
  else
    find "$ROOT" -type f -printf '%h\n' | sort | uniq -c | awk '{print $2 "," $1}' | while IFS=',' read -r folder count; do
      counts["$folder"]=$count
    done
  fi
else
  # Non-recursive: count in immediate subfolders
  for d in "$ROOT"/*/; do
    if [ -d "$d" ]; then
      folder="${d%/}"  # Remove trailing /
      if [ ${#EXTENSIONS[@]} -gt 0 ]; then
        names=$(build_names)
        count=$(find "$d" -maxdepth 1 -type f \( $names \) | wc -l)
      else
        count=$(find "$d" -maxdepth 1 -type f | wc -l)
      fi
      counts["$folder"]=$count
    fi
  done
fi

# Output
output() {
  if [ "$FORMAT" = "json" ]; then
    echo "{"
    first=true
    for folder in $(printf '%s\n' "${!counts[@]}" | sort); do
      if [ "$first" = true ]; then
        first=false
      else
        echo ","
      fi
      echo -n "  \"$folder\": ${counts[$folder]}"
    done
    echo ""
    echo "}"
  elif [ "$FORMAT" = "csv" ]; then
    echo "folder,count"
    for folder in $(printf '%s\n' "${!counts[@]}" | sort); do
      echo "$folder,${counts[$folder]}"
    done
  else
    for folder in $(printf '%s\n' "${!counts[@]}" | sort); do
      echo "$folder: ${counts[$folder]}"
    done
  fi
}

if [ -n "$OUTPUT" ]; then
  output > "$OUTPUT"
  echo "Output written to $OUTPUT"
else
  output
fi