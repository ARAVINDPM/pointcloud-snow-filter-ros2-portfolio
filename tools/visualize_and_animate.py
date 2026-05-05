"""
Interactive visualization and GIF animation of filtering pipeline.

Generates before/after filtering visualizations and creates animated GIFs
showing the effect of each filter on contaminated point clouds.

Usage:
    python visualize_and_animate.py <input_pcd> [--output_dir results/]
    python visualize_and_animate.py data/synthetic_snow_scans/mannequin_000_snow.pcd
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import logging
import argparse
import numpy as np
import open3d as o3d

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from lidar_snow_filter.filters import LiDARFilters
from lidar_snow_filter.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PointCloudVisualizer:
    """Render and animate point clouds with matplotlib."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir or RESULTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_to_image(self, pcd: o3d.geometry.PointCloud,
                       title: str = "", width: int = 800, height: int = 600) -> np.ndarray:
        """
        Render point cloud to numpy array image.

        Args:
            pcd: Open3D point cloud
            title: Title for visualization
            width, height: Image dimensions (approximate)

        Returns:
            RGB image as numpy array
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.error("matplotlib required for visualization: pip install matplotlib")
            return None

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        points = np.asarray(pcd.points)

        # Color by density (Z-axis for height)
        if len(points) > 0:
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            colors = (points[:, 2] - z_min) / (z_max - z_min) if z_max > z_min else np.zeros(len(points))
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=colors, cmap='viridis', s=1, alpha=0.6)

        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Set equal aspect ratio and limits
        if len(points) > 0:
            max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                                  points[:, 1].max()-points[:, 1].min(),
                                  points[:, 2].max()-points[:, 2].min()]).max() / 2.0
            mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Convert figure to image using in-memory buffer
        from io import BytesIO
        from PIL import Image as PILImage

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        try:
            img = PILImage.open(buf)
            image = np.array(img.convert('RGB'))
        except Exception as e:
            logger.error(f"Failed to render figure: {e}")
            plt.close(fig)
            return None
        finally:
            buf.close()

        plt.close(fig)
        return image

    def create_comparison_animation(self, input_pcd: o3d.geometry.PointCloud,
                                   output_name: str = "filtering_demo"):
        """
        Create side-by-side animation of input → filtered clouds.

        Args:
            input_pcd: Contaminated point cloud
            output_name: Output filename (without extension)
        """
        if not HAS_IMAGEIO:
            logger.warning("imageio required for GIF generation: pip install imageio")
            return False

        logger.info(f"Generating comparison animation: {output_name}")

        frames = []
        filters_list = [
            ('Input (Snow Contaminated)', input_pcd),
            ('SOR Filter', LiDARFilters.sor(input_pcd)[0]),
            ('ROR Filter', LiDARFilters.ror(input_pcd)[0]),
            ('DSOR Filter (Height-Adaptive)', LiDARFilters.dsor(input_pcd)[0]),
            ('DROR Filter (Sector-Based)', LiDARFilters.dror(input_pcd)[0]),
        ]

        # Render each stage
        for label, pcd in filters_list:
            logger.info(f"  Rendering: {label}")
            image = self.render_to_image(pcd, title=label)
            if image is not None:
                frames.append(image)

        if not frames:
            logger.error("No frames generated")
            return False

        # Normalize frame sizes
        from PIL import Image as PILImage
        target_height, target_width = frames[0].shape[:2]
        normalized_frames = []
        for frame in frames:
            if frame.shape[:2] != (target_height, target_width):
                img = PILImage.fromarray(frame)
                img = img.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
                normalized_frames.append(np.array(img))
            else:
                normalized_frames.append(frame)

        # Save as GIF (loop animation)
        output_path = self.output_dir / f"{output_name}.gif"
        logger.info(f"Saving GIF to: {output_path}")

        # Repeat frames for smooth loop
        frames_loop = normalized_frames + normalized_frames[::-1]  # Add reverse for smooth loop
        imageio.mimsave(str(output_path), frames_loop, duration=0.5, loop=0)
        logger.info(f"✓ GIF created: {output_path}")

        return True

    def create_filter_comparison_chart(self, input_pcd: o3d.geometry.PointCloud):
        """
        Create side-by-side comparison of all filters.

        Args:
            input_pcd: Contaminated point cloud
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.error("matplotlib required")
            return False

        logger.info("Creating filter comparison chart")

        filters_dict = {
            'Input\n(Contaminated)': input_pcd,
            'SOR': LiDARFilters.sor(input_pcd)[0],
            'ROR': LiDARFilters.ror(input_pcd)[0],
            'DSOR': LiDARFilters.dsor(input_pcd)[0],
            'DROR': LiDARFilters.dror(input_pcd)[0],
        }

        fig = plt.figure(figsize=(18, 12))

        for idx, (title, pcd) in enumerate(filters_dict.items(), 1):
            ax = fig.add_subplot(2, 3, idx, projection='3d')

            points = np.asarray(pcd.points)
            point_count = len(points)

            if len(points) > 0:
                z_vals = points[:, 2]
                z_min, z_max = z_vals.min(), z_vals.max()
                colors = (z_vals - z_min) / (z_max - z_min) if z_max > z_min else np.zeros(len(points))

                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=colors, cmap='viridis', s=1, alpha=0.7)

                # Set limits
                max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                                      points[:, 1].max()-points[:, 1].min(),
                                      points[:, 2].max()-points[:, 2].min()]).max() / 2.0
                mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
                mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
                mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            ax.set_title(f"{title}\n({point_count} pts)", fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=7)

        plt.suptitle("LiDAR Filter Comparison: Snow Removal",
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        output_path = self.output_dir / "filter_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved comparison chart: {output_path}")
        plt.close()

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and animate LiDAR filtering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Input PCD file (snow-contaminated)")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR),
                       help="Output directory for visualizations")
    parser.add_argument("--animate", action="store_true", default=True,
                       help="Generate animated GIF (requires imageio)")
    parser.add_argument("--chart", action="store_true", default=True,
                       help="Generate comparison chart")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    # Load point cloud
    logger.info(f"Loading: {input_path}")
    pcd = o3d.io.read_point_cloud(str(input_path))
    logger.info(f"Points: {len(pcd.points)}")

    # Create visualizer
    vis = PointCloudVisualizer(args.output_dir)

    # Generate visualization
    if args.animate:
        vis.create_comparison_animation(pcd, output_name=input_path.stem)

    if args.chart:
        vis.create_filter_comparison_chart(pcd)

    logger.info("✓ Visualization complete!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
