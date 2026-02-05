"""
Background Removal Script for Student Photos
Uses rembg (U2Net deep learning model) for precise background removal

Installation:
    pip install rembg pillow

Usage:
    python remove_backgrounds.py
"""

import os
from pathlib import Path

try:
    from rembg import remove
    from PIL import Image
except ImportError:
    print("Required packages not installed. Please run:")
    print("    pip install rembg pillow")
    print("\nThis will also download the U2Net model (~170MB) on first run.")
    exit(1)


def remove_background(input_path, output_path):
    """Remove background from a single image with high precision."""
    try:
        # Open the input image
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()

        # Remove background using U2Net model
        output_data = remove(
            input_data,
            alpha_matting=True,  # Enable alpha matting for better edge precision
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )

        # Save the output image
        with open(output_path, 'wb') as output_file:
            output_file.write(output_data)

        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    # Path to student photos
    students_dir = Path(__file__).parent / "assets" / "students"

    if not students_dir.exists():
        print(f"Error: Directory not found: {students_dir}")
        return

    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    # Get all image files (excluding already processed _nobg files)
    image_files = [
        f for f in students_dir.iterdir()
        if f.suffix.lower() in image_extensions
        and '_nobg' not in f.stem
    ]

    if not image_files:
        print("No images found to process.")
        return

    print(f"Found {len(image_files)} images to process")
    print("=" * 50)
    print("Using rembg with U2Net model + alpha matting")
    print("=" * 50)
    print()

    successful = 0
    failed = 0
    skipped = 0

    for i, image_path in enumerate(sorted(image_files), 1):
        # Output filename with _nobg suffix
        output_path = students_dir / f"{image_path.stem}_nobg.png"

        # Check if already processed
        if output_path.exists():
            print(f"[{i}/{len(image_files)}] Skipping {image_path.name} (already processed)")
            skipped += 1
            continue

        print(f"[{i}/{len(image_files)}] Processing {image_path.name}...", end=" ")

        if remove_background(image_path, output_path):
            print("Done!")
            successful += 1
        else:
            print("Failed!")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {successful} successful, {skipped} skipped, {failed} failed")
    print("=" * 50)

    if successful > 0:
        print("\nNext step: Update the HTML to use the new _nobg.png images")


if __name__ == "__main__":
    main()
