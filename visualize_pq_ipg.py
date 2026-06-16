"""
PQ-IPG Visualization Script
=============================
Visualize Pose Quality-aware IPG weights for standard pose skeletons.

This script:
1. Loads the 8 standard pose skeleton images from IPG/standard_poses/
2. Computes pose quality scores using skeletal density, body coverage, and symmetry
3. Normalizes to weights via softmax
4. Generates a visualization grid showing each pose with its weight

Usage:
    python visualize_pq_ipg.py [--pose_dir IPG/standard_poses] [--output viz_pq_ipg.png]
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'demo/TransReID-main'))
from utils.pose_quality import (
    compute_pose_quality_from_image,
    compute_pq_ipg_weights,
)


def create_visualization(pose_dir, output_path, alpha=0.7, beta=0.3):
    """
    Create a visualization grid showing each standard pose with its PQ-IPG weight.

    Args:
        pose_dir: Directory containing standard pose images (1.jpg .. N.jpg)
        output_path: Output image path
        alpha: Skeletal density weight
        beta: Body coverage weight
    """
    if not os.path.exists(pose_dir):
        print(f"Error: Pose directory '{pose_dir}' not found.")
        return

    # Load pose images
    pose_files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    )
    if len(pose_files) == 0:
        print(f"Error: No image files found in '{pose_dir}'.")
        return

    print(f"Found {len(pose_files)} standard poses in {pose_dir}")

    pose_images = []
    qualities = []
    for f in pose_files:
        img = Image.open(os.path.join(pose_dir, f)).convert("RGB")
        pose_images.append(img)
        q = compute_pose_quality_from_image(np.array(img), alpha=alpha, beta=beta)
        qualities.append(q)
        print(f"  {f}: quality = {q:.6f}")

    # Compute weights
    weights = compute_pq_ipg_weights(qualities)

    # Summary stats
    print("\n" + "=" * 60)
    print("PQ-IPG Weight Summary:")
    print("=" * 60)
    for i, (f, q, w) in enumerate(zip(pose_files, qualities, weights.tolist())):
        print(f"  Pose {i+1} ({f}): quality={q:.4f}, weight={w:.4f}")
    print(f"\n  Equal weight baseline: {1/len(pose_files):.4f} per pose")
    print(f"  Max weight: {weights.max():.4f} (Pose {weights.argmax().item()+1})")
    print(f"  Min weight: {weights.min():.4f} (Pose {weights.argmin().item()+1})")
    print(f"  Weight range: {weights.max().item() - weights.min().item():.4f}")

    # Create visualization grid
    grid_cols = len(pose_files)
    grid_rows = 2  # top row: poses, bottom row: weights + bars

    # Resize poses to uniform size
    pose_w, pose_h = 120, 160
    resized_poses = [img.resize((pose_w, pose_h)) for img in pose_images]

    # Create canvas
    padding = 20
    bar_h = 40
    label_h = 30
    cell_h = pose_h + bar_h + label_h + padding * 2
    cell_w = pose_w + padding * 2
    canvas_w = cell_w * grid_cols
    canvas_h = cell_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Try to use a nice font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_title = ImageFont.load_default()

    # Draw title
    title = f"PQ-IPG: Pose Quality-Aware Weights  (alpha={alpha}, beta={beta})"
    draw.text(((canvas_w - draw.textlength(title, font=font_title)) // 2, 2),
              title, fill="black", font=font_title)

    # Draw each pose and its weight
    for i in range(len(pose_files)):
        x_center = i * cell_w + cell_w // 2
        x_offset = i * cell_w + padding
        y_offset = padding + 20

        # Paste pose image
        canvas.paste(resized_poses[i], (x_offset, y_offset))

        # Draw pose number
        draw.text((x_center - 10, y_offset + pose_h + 4),
                  f"Pose {i+1}", fill="black", font=font)

        # Draw weight value
        w_text = f"w = {weights[i]:.4f}"
        tw = draw.textlength(w_text, font=font_small)
        draw.text((x_center - tw // 2, y_offset + pose_h + label_h),
                  w_text, fill="black", font=font_small)

        # Draw quality bar
        bar_y = y_offset + pose_h + label_h + 20
        bar_width = pose_w
        bar_height = 10

        # Background
        draw.rectangle([x_offset, bar_y, x_offset + bar_width, bar_y + bar_height],
                       fill="lightgray", outline="gray")

        # Filled portion (green = high weight, red = low weight)
        fill_w = int(bar_width * weights[i].item())
        r = int(255 * (1 - weights[i].item()))
        g = int(255 * weights[i].item())
        bar_color = (r, g, 0)
        draw.rectangle([x_offset, bar_y, x_offset + fill_w, bar_y + bar_height],
                       fill=bar_color)

        # Draw equal-weight baseline
        baseline_x = x_offset + int(bar_width / len(pose_files))
        draw.line([(baseline_x, bar_y - 2), (baseline_x, bar_y + bar_height + 2)],
                  fill="blue", width=2)

        # Draw border around pose
        draw.rectangle([x_offset, y_offset, x_offset + pose_w, y_offset + pose_h],
                       outline="gray")

    # Add legend
    legend_y = canvas_h - 15
    legend_text = "Green = high weight (better pose), Blue dashed = equal-weight baseline"
    lt = draw.textlength(legend_text, font=font_small)
    draw.text(((canvas_w - lt) // 2, legend_y),
              legend_text, fill="gray", font=font_small)

    canvas.save(output_path)
    print(f"\nVisualization saved to: {output_path}")


def analyze_keypoint_quality(keypoint_path, alpha=0.7, beta=0.3):
    """
    Analyze keypoint quality statistics from VeRi keypoint annotation files.

    Args:
        keypoint_path: Path to keypoint_train.txt or keypoint_test.txt
        alpha: Visibility rate weight
        beta: Spatial coverage weight
    """
    from utils.pose_quality import compute_pose_quality_from_keypoints

    with open(keypoint_path, 'r') as f:
        lines = f.readlines()

    qualities = []
    visible_keypoints = []

    print(f"\nAnalyzing keypoint quality from: {keypoint_path}")
    print(f"Total samples: {len(lines)}")
    print("=" * 60)

    for line in lines:
        parts = line.strip().split(' ')
        img_path = parts[0]
        # 20 keypoints = 40 coordinates
        kp_values = list(map(int, parts[1:41]))
        keypoints = np.array(kp_values).reshape(-1, 2)

        q = compute_pose_quality_from_keypoints(keypoints, alpha=alpha, beta=beta)
        qualities.append(q)

        visible = ((keypoints[:, 0] != -1) & (keypoints[:, 1] != -1)).sum()
        visible_keypoints.append(visible)

    qualities = np.array(qualities)
    visible_keypoints = np.array(visible_keypoints)

    print(f"  Mean quality: {qualities.mean():.4f} +/- {qualities.std():.4f}")
    print(f"  Min quality: {qualities.min():.4f}, Max quality: {qualities.max():.4f}")
    print(f"  Mean visible keypoints: {visible_keypoints.mean():.1f} / 20")
    print(f"  Samples with < 5 keypoints: {(visible_keypoints < 5).sum()} "
          f"({100 * (visible_keypoints < 5).sum() / len(visible_keypoints):.1f}%)")
    print(f"  Samples with >= 15 keypoints: {(visible_keypoints >= 15).sum()} "
          f"({100 * (visible_keypoints >= 15).sum() / len(visible_keypoints):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PQ-IPG Visualization")
    parser.add_argument("--pose_dir", type=str, default="IPG/standard_poses",
                        help="Directory with standard pose images")
    parser.add_argument("--output", type=str, default="viz_pq_ipg.png",
                        help="Output visualization path")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Skeletal density weight")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Body coverage weight")
    parser.add_argument("--keypoint_file", type=str, default=None,
                        help="Analyze keypoint quality from .txt file (optional)")

    args = parser.parse_args()

    # Create standard pose visualization
    create_visualization(args.pose_dir, args.output, args.alpha, args.beta)

    # Optionally analyze keypoint quality
    if args.keypoint_file is not None:
        analyze_keypoint_quality(args.keypoint_file, args.alpha, args.beta)
