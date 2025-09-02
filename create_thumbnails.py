#!/usr/bin/env python3
import os
import argparse
from PIL import Image

def generate_thumbnail(source_path, dest_path, size):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with Image.open(source_path) as img:
        img.thumbnail(size)
        img.save(dest_path, "JPEG")

def main():
    parser = argparse.ArgumentParser(description="Generate thumbnails for all images in storage directory.")
    parser.add_argument('--root', default='storage', help='Root directory containing user folders')
    parser.add_argument('--size', default='200,200', help='Thumbnail size as WIDTH,HEIGHT (e.g., 200,200)')
    args = parser.parse_args()

    try:
        width, height = map(int, args.size.split(','))
    except ValueError:
        print("Invalid size format. Use WIDTH,HEIGHT (e.g., 200,200)")
        return

    for user in os.listdir(args.root):
        truths_dir = os.path.join(args.root, user, 'truths')
        thumbs_dir = os.path.join(args.root, user, 'thumbs')
        if not os.path.isdir(truths_dir):
            continue

        for filename in os.listdir(truths_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                source_path = os.path.join(truths_dir, filename)
                name, _ = os.path.splitext(filename)
                dest_path = os.path.join(thumbs_dir, f"{name}_thumb.jpeg")
                generate_thumbnail(source_path, dest_path, (width, height))
                print(f"Generated thumbnail for {source_path} -> {dest_path}")

if __name__ == '__main__':
    main()
