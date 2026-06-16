"""
Pose Quality-Aware IPG (PQ-IPG)
=================================
Compute quality scores for pose skeleton images used in IPG generation.

Key idea:
  Not all generated pose variants have equal quality. Poses with more
  visible body parts (full skeleton, frontal view) tend to generate
  better images with richer identity information.

Quality = α · skeletal_density + β · body_coverage

where:
  - skeletal_density: fraction of non-black pixels in the skeleton image
  - body_coverage: fraction of image height covered by the skeleton
  - α, β: weighting coefficients (default α=0.7, β=0.3)

The raw quality scores are normalized via softmax to get final weights.
"""

import numpy as np
from PIL import Image
import torch


def compute_pose_quality_from_image(pose_img: np.ndarray, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3) -> float:
    """
    Compute pose quality from a pose skeleton image.

    Uses regional analysis of the skeleton image to evaluate pose quality:
    - Head/content-rich poses (frontal, face visible) get higher scores
    - Asymmetric side/back views get lower scores

    Args:
        pose_img: Pose skeleton image as numpy array (H, W, 3) or (H, W)
        alpha: Weight for head/upper-body prominence
        beta: Weight for body coverage (height span)
        gamma: Weight for symmetry (frontal vs side view)

    Returns:
        Quality score (float)
    """
    if pose_img.ndim == 3:
        gray = np.mean(pose_img, axis=2)
    else:
        gray = pose_img

    H, W = gray.shape
    skeleton_mask = gray > 10

    # 1. Head prominence: skeleton content in the top 1/8 of image
    #    Higher head content = face visible = frontal view = better for ReID
    head_region = skeleton_mask[:H // 8, :]
    head_prominence = head_region.sum() / head_region.size if head_region.size > 0 else 0.0

    # 2. Upper body content: top 1/3
    upper_region = skeleton_mask[:H // 3, :]
    upper_content = upper_region.sum() / upper_region.size if upper_region.size > 0 else 0.0

    # 3. Body coverage: vertical span of the skeleton
    row_sum = skeleton_mask.sum(axis=1)
    rows_with_content = np.where(row_sum > 0)[0]
    if len(rows_with_content) > 0:
        body_coverage = (rows_with_content[-1] - rows_with_content[0]) / H
    else:
        body_coverage = 0.0

    # 4. Symmetry: difference between left and right halves
    #    Symmetric = frontal view, asymmetric = side/back view
    left_half = skeleton_mask[:, :W // 2]
    right_half = np.fliplr(skeleton_mask[:, W // 2:])
    # Ensure same shape by truncating if necessary
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w > 0:
        symmetry = (left_half[:, :min_w] == right_half[:, :min_w]).sum() / (H * min_w)
    else:
        symmetry = 0.0

    # 5. Balance penalty: large left-right asymmetry means side view
    left_density = skeleton_mask[:, :W // 2].sum()
    right_density = skeleton_mask[:, W // 2:].sum()
    total = left_density + right_density
    if total > 0:
        balance = 1.0 - abs(left_density - right_density) / total
    else:
        balance = 0.0

    # Quality = head prominence + upper body + coverage + symmetry + balance
    quality = (
        alpha * (0.5 * head_prominence + 0.5 * upper_content)  # head/upper body
        + beta * body_coverage                                  # full body visible
        + gamma * (0.5 * symmetry + 0.5 * balance)             # frontal view bonus
    )

    return float(quality)


def compute_pose_quality_from_keypoints(keypoints: np.ndarray, alpha: float = 0.7, beta: float = 0.3) -> float:
    """
    Compute pose quality from keypoint coordinates.

    Args:
        keypoints: Array of shape (K, 2) where K=20 keypoints.
                   (-1, -1) indicates missing/occluded keypoint.
        alpha: Weight for keypoint visibility rate
        beta: Weight for spatial coverage

    Returns:
        Quality score (float)
    """
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)

    K = len(keypoints)

    # Visibility rate: fraction of keypoints that are visible
    visible_mask = (keypoints[:, 0] != -1) & (keypoints[:, 1] != -1)
    visible_rate = visible_mask.sum() / K

    # Spatial coverage: how spread out the visible keypoints are
    if visible_mask.sum() >= 2:
        visible_kpts = keypoints[visible_mask]
        x_span = visible_kpts[:, 0].max() - visible_kpts[:, 0].min()
        y_span = visible_kpts[:, 1].max() - visible_kpts[:, 1].min()
        # Normalize by image dimensions (assume 256x128 for ReID)
        spatial_coverage = 0.5 * (x_span / 128 + y_span / 256)
        spatial_coverage = min(spatial_coverage, 1.0)
    else:
        spatial_coverage = 0.0

    # Upper body keypoints (typically more informative for ReID)
    upper_body_indices = list(range(11))  # nose, eyes, ears, shoulders, elbows, wrists
    upper_visible = sum(visible_mask[i] for i in upper_body_indices if i < K) / len(upper_body_indices)

    quality = alpha * visible_rate + beta * spatial_coverage + 0.2 * upper_visible

    return float(quality)


def compute_pq_ipg_weights(pose_qualities: list, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalize pose quality scores into weights via softmax.

    Args:
        pose_qualities: List of quality scores, one per pose variant
        temperature: Softmax temperature (lower = sharper weights)

    Returns:
        Weight tensor with shape (N,) summing to 1
    """
    q = torch.tensor(pose_qualities, dtype=torch.float32)
    weights = torch.softmax(q / temperature, dim=0)
    return weights


def precompute_standard_pose_weights(pose_dir: str = "standard_poses", alpha: float = 0.7, beta: float = 0.3) -> list:
    """
    Pre-compute quality weights for standard pose images used in IPG.

    Args:
        pose_dir: Directory containing standard pose images (1.jpg .. N.jpg)
        alpha: Weight for skeletal density
        beta: Weight for body coverage

    Returns:
        List of (pose_index, weight) tuples sorted by pose index
    """
    import os
    from PIL import Image

    pose_files = sorted(os.listdir(pose_dir))
    qualities = []

    for f in pose_files:
        img = Image.open(os.path.join(pose_dir, f)).convert("RGB")
        img_arr = np.array(img)
        q = compute_pose_quality_from_image(img_arr, alpha=alpha, beta=beta)
        qualities.append(q)

    # Normalize via softmax
    weights = torch.softmax(torch.tensor(qualities), dim=0).tolist()

    result = []
    for i, (f, w) in enumerate(zip(sorted(pose_files), weights)):
        result.append((i, float(w)))
        print(f"  {f}: quality={qualities[i]:.4f}, weight={w:.4f}")

    return result


if __name__ == "__main__":
    # Demo: compute weights for standard poses
    import os
    import sys

    # Try both possible paths
    for pose_dir in ["IPG/standard_poses", "standard_poses", "../standard_poses"]:
        if os.path.exists(pose_dir):
            print(f"\nPre-computing PQ-IPG weights from {pose_dir}/:")
            print("=" * 50)
            weights = precompute_standard_pose_weights(pose_dir)
            print("\nFinal PQ-IPG weights (sorted by weight):")
            print("=" * 50)
            for idx, wgt in sorted(weights, key=lambda x: x[1], reverse=True):
                print(f"  Pose {idx + 1}: {wgt:.4f}")
            break
    else:
        print("No standard_poses directory found. Run from project root.")
