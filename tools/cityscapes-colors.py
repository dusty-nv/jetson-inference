#!/usr/bin/env python3
#
# Generates classes.txt / colors.txt for the FCN-ResNet18 Cityscapes models
# distributed by jetson-inference, using the official per-class RGB values
# from the Cityscapes dataset (cityscapesScripts/helpers/labels.py upstream:
# https://github.com/mcordts/cityscapesScripts).
#
# Why this exists: colors.txt inside the released model tarballs
# (e.g. FCN-ResNet18-Cityscapes-512x256.tar.gz) was hand-edited at some point
# and drifted from the official palette -- two rows ended up as byte-for-byte
# duplicates of other rows (ego_vehicle == road, traffic_sign == traffic_light),
# and sidewalk was set to a color that isn't in the official palette at all.
# This script lets classes.txt/colors.txt be regenerated deterministically
# instead of hand-maintained, and prints a warning if it ever finds two
# classes sharing an identical color.
#
# Usage:
#   python3 cityscapes-colors.py path/to/classes.txt
#
#   This reads the existing classes.txt (one class name per line, in the
#   exact order used by the model) and writes a colors.txt next to it with
#   the correct RGB triplet for each class name, in the same order.
#
# If a class name in classes.txt isn't found in the official palette,
# the script stops and reports which line/name it couldn't resolve rather
# than silently emitting a wrong color.

import argparse
import os
import sys

# name -> (R, G, B), taken from the official Cityscapes label definitions.
# jetson-inference's classes.txt uses underscores ("traffic_light") where
# upstream Cityscapes uses spaces ("traffic light"), so both forms are
# registered below.
CITYSCAPES_COLORS = {
    "unlabeled":            (0, 0, 0),
    "void":                 (0, 0, 0),
    "ego vehicle":          (0, 0, 0),
    "ego_vehicle":          (0, 0, 0),
    "rectification border": (0, 0, 0),
    "out of roi":           (0, 0, 0),
    "static":               (0, 0, 0),
    "dynamic":              (111, 74, 0),
    "ground":               (81, 0, 81),
    "road":                 (128, 64, 128),
    "sidewalk":             (244, 35, 232),
    "parking":              (250, 170, 160),
    "rail track":           (230, 150, 140),
    "building":             (70, 70, 70),
    "wall":                 (102, 102, 156),
    "fence":                (190, 153, 153),
    "guard rail":           (180, 165, 180),
    "bridge":               (150, 100, 100),
    "tunnel":               (150, 120, 90),
    "pole":                 (153, 153, 153),
    "polegroup":            (153, 153, 153),
    "traffic light":        (250, 170, 30),
    "traffic_light":        (250, 170, 30),
    "traffic sign":         (220, 220, 0),
    "traffic_sign":         (220, 220, 0),
    "vegetation":           (107, 142, 35),
    "terrain":              (152, 251, 152),
    "sky":                  (70, 130, 180),
    "person":               (220, 20, 60),
    "rider":                (255, 0, 0),
    "car":                  (0, 0, 142),
    "truck":                (0, 0, 70),
    "bus":                  (0, 60, 100),
    "caravan":              (0, 0, 90),
    "trailer":              (0, 0, 110),
    "train":                (0, 80, 100),
    "motorcycle":           (0, 0, 230),
    "bicycle":              (119, 11, 32),
    "license plate":        (0, 0, 142),
}


def load_classes(classes_path):
    with open(classes_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_color(name):
    key = name.strip().lower()
    if key in CITYSCAPES_COLORS:
        return CITYSCAPES_COLORS[key]
    # be a little forgiving of alternate separators
    alt = key.replace("_", " ")
    if alt in CITYSCAPES_COLORS:
        return CITYSCAPES_COLORS[alt]
    return None


def generate_colors(classes):
    colors = []
    unresolved = []
    for i, name in enumerate(classes):
        color = resolve_color(name)
        if color is None:
            unresolved.append((i, name))
        else:
            colors.append(color)
    return colors, unresolved


def check_duplicates(classes, colors):
    seen = {}
    duplicates = []
    for name, color in zip(classes, colors):
        if color in seen:
            duplicates.append((seen[color], name, color))
        else:
            seen[color] = name
    return duplicates


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate colors.txt for a Cityscapes classes.txt "
                     "using the official Cityscapes RGB palette."
    )
    parser.add_argument(
        "classes_file",
        help="path to the existing classes.txt for the model",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="path to write colors.txt to (default: colors.txt next to classes_file)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.classes_file):
        print(f"error: '{args.classes_file}' not found", file=sys.stderr)
        sys.exit(1)

    classes = load_classes(args.classes_file)
    colors, unresolved = generate_colors(classes)

    if unresolved:
        print("error: could not resolve official color for the following classes:", file=sys.stderr)
        for i, name in unresolved:
            print(f"  line {i + 1}: '{name}'", file=sys.stderr)
        print("\nAdd these class names to CITYSCAPES_COLORS in this script and re-run.", file=sys.stderr)
        sys.exit(1)

    duplicates = check_duplicates(classes, colors)
    if duplicates:
        print("warning: the following classes ended up with identical colors:", file=sys.stderr)
        for existing_name, name, color in duplicates:
            print(f"  '{name}' matches '{existing_name}' -> {color}", file=sys.stderr)
        print("(this can be legitimate for void-type classes, but double check it's intentional)\n", file=sys.stderr)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.classes_file)), "colors.txt")

    with open(output_path, "w") as f:
        for color in colors:
            f.write(f"{color[0]} {color[1]} {color[2]}\n")

    print(f"wrote {len(colors)} colors to '{output_path}'")


if __name__ == "__main__":
    main()
