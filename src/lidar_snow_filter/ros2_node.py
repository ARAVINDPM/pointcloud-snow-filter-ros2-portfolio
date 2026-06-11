"""Minimal ROS2 node wrapping the snow filters.

Subscribes to a ``sensor_msgs/msg/PointCloud2`` topic, applies one of the
package's filters (SOR / ROR / DSOR / DROR), and republishes the filtered
cloud. Keeps the original message header so downstream TF lookups still work.

Run (requires a ROS2 installation, e.g. Humble):

    ros2 run lidar_snow_filter snow_filter_node
    # or directly:
    python -m lidar_snow_filter.ros2_node --ros-args -p filter:=ror

Parameters:
    filter (str): one of "sor", "ror", "dsor", "dror". Default "ror".
    input_topic (str): default "/points_raw".
    output_topic (str): default "/points_filtered".

ROS2 is an optional dependency: this module imports rclpy at call time so
the rest of the package works without ROS2 installed.
"""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np
import open3d as o3d

from .filters import LiDARFilters

_FILTERS = ("sor", "ror", "dsor", "dror")


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """Extract an Nx3 float array from a sensor_msgs/msg/PointCloud2.

    Assumes x, y, z are float32 fields (the common case for LiDAR drivers).
    Non-finite points are dropped.
    """
    field_offsets = {f.name: f.offset for f in msg.fields}
    for axis in ("x", "y", "z"):
        if axis not in field_offsets:
            raise ValueError(f"PointCloud2 message missing '{axis}' field")

    n = msg.width * msg.height
    step = msg.point_step
    data = bytes(msg.data)
    xyz = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        base = i * step
        for j, axis in enumerate(("x", "y", "z")):
            off = base + field_offsets[axis]
            xyz[i, j] = struct.unpack_from("<f", data, off)[0]
    return xyz[np.isfinite(xyz).all(axis=1)]


def xyz_to_pointcloud2(xyz: np.ndarray, header, pointcloud2_cls, pointfield_cls):
    """Pack an Nx3 array into a sensor_msgs/msg/PointCloud2 (xyz float32)."""
    msg = pointcloud2_cls()
    msg.header = header
    msg.height = 1
    msg.width = int(len(xyz))
    msg.fields = [
        pointfield_cls(name=n, offset=4 * i, datatype=7, count=1)  # 7 = FLOAT32
        for i, n in enumerate(("x", "y", "z"))
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * msg.width
    msg.is_dense = True
    msg.data = np.ascontiguousarray(xyz, dtype=np.float32).tobytes()
    return msg


def filter_xyz(xyz: np.ndarray, filter_name: str) -> Tuple[np.ndarray, dict]:
    """Apply a named filter to an Nx3 array. Pure function, ROS-free."""
    if filter_name not in _FILTERS:
        raise ValueError(f"filter must be one of {_FILTERS}, got {filter_name!r}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    filtered, meta = getattr(LiDARFilters, filter_name)(pcd)
    return np.asarray(filtered.points), meta


def main(args=None):  # pragma: no cover - requires a ROS2 install
    """Entry point. Imports rclpy lazily so ROS2 stays optional."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField

    class SnowFilterNode(Node):
        def __init__(self):
            super().__init__("snow_filter_node")
            self.declare_parameter("filter", "ror")
            self.declare_parameter("input_topic", "/points_raw")
            self.declare_parameter("output_topic", "/points_filtered")

            self.filter_name = str(self.get_parameter("filter").value).lower()
            if self.filter_name not in _FILTERS:
                raise ValueError(f"filter param must be one of {_FILTERS}")

            in_topic = str(self.get_parameter("input_topic").value)
            out_topic = str(self.get_parameter("output_topic").value)
            self._pub = self.create_publisher(PointCloud2, out_topic, 10)
            self._sub = self.create_subscription(PointCloud2, in_topic, self._on_cloud, 10)
            self._pc2, self._pf = PointCloud2, PointField
            self.get_logger().info(
                f"snow_filter_node: {self.filter_name.upper()} {in_topic} -> {out_topic}"
            )

        def _on_cloud(self, msg):
            try:
                xyz = pointcloud2_to_xyz(msg)
                if len(xyz) == 0:
                    return
                filtered, meta = filter_xyz(xyz, self.filter_name)
                self._pub.publish(
                    xyz_to_pointcloud2(filtered, msg.header, self._pc2, self._pf)
                )
                self.get_logger().debug(
                    f"{meta['input_points']} -> {meta['output_points']} pts"
                )
            except Exception as exc:  # never kill the node on a bad frame
                self.get_logger().error(f"filtering failed: {exc}")

    rclpy.init(args=args)
    node = SnowFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
