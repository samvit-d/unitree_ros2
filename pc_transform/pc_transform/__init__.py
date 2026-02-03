import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs_py import point_cloud2
import numpy as np

import threading
from concurrent.futures import ThreadPoolExecutor


class PCTransform(Node):
    """A ROS2 Node that transforms and slices the Unitree L2 point cloud."""

    def __init__(self):
        super().__init__('pc_transform')
        
        # params
        self.max_points = 100000
        self.height_filter_min = -2.0
        self.height_filter_max = 0.5
        self.ransac_iterations = 100
        self.ransac_threshold = 0.02
        self.auto_estimate_interval = 5.0
        self.min_points = 1000
        
        # Accumulated points
        self.accumulated_points = []
        self.total_points = 0
        self.clouds_received = 0
        
        # Multithreading
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.estimation_running = False
        
        # PointCloud subscription
        self.pc_sub = self.create_subscription(
            PointCloud2, 
            '/utlidar/cloud',
            self.cloud_callback,
            10
        )
        
        # Publish plane parameters
        self.plane_pub = self.create_publisher(
            Float64MultiArray,
            '/floor_plane/estimate',
            10
        )

        # RViz visualization publishers
        self.plane_marker_pub = self.create_publisher(
            Marker,
            '/floor_plane/marker',
            10
        )
        self.normal_marker_pub = self.create_publisher(
            Marker,
            '/floor_plane/normal',
            10
        )
        
        # Timer for periodic estimation
        self.estimate_timer = self.create_timer(
            self.auto_estimate_interval,
            self.run_estimation
        )
        

    def cloud_callback(self, msg: PointCloud2):
        """Method that is periodically called by the timer."""
        # Overflow
        with self.lock:
            if self.total_points >= self.max_points:
                return
        
        # Read points (returns structured array)
        pts_structured = np.array(list(point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )))
        
        if len(pts_structured) == 0:
            return
            
        # Convert structured array to regular Nx3 array
        points = np.column_stack([
            pts_structured['x'],
            pts_structured['y'],
            pts_structured['z']
        ]).astype(np.float32)

        # Downsample if we're getting too many points
        if self.total_points + len(points) > self.max_points:
            keep = self.max_points - self.total_points
            indices = np.random.choice(len(points), keep, replace=False)
            points = points[indices]

        with self.lock:
            self.accumulated_points.append(points)
            self.total_points += len(points)

        self.clouds_received += 1

        if self.clouds_received % 100 == 0:
            self.get_logger().info(
                f'Accumulated {self.total_points} floor candidate points '
                f'from {self.clouds_received} clouds'
            )
            
    def fit_plane_ransac(self, points: np.ndarray):
        """
        Fit a plane to points using RANSAC.
        
        Returns:
            (plane_params, inlier_mask) where plane_params is [a, b, c, d] 
            for plane equation ax + by + cz + d = 0, with [a,b,c] as unit normal.
        """
        n_points = len(points)
        if n_points < 3:
            return None, None

        best_inliers = None
        best_n_inliers = 0
        best_plane = None

        for _ in range(self.ransac_iterations):
            # Random sample of 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-10:
                continue

            normal = normal / norm

            # Ensure normal points "up" (positive z component)
            if normal[2] < 0:
                normal = -normal

            # Plane equation: normal . (p - p1) = 0
            # ax + by + cz + d = 0 where d = -normal . p1
            d = -np.dot(normal, p1)

            # Compute distances to plane
            distances = np.abs(points @ normal + d)

            # Count inliers
            inlier_mask = distances < self.ransac_threshold
            n_inliers = np.sum(inlier_mask)

            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inliers = inlier_mask
                best_plane = np.array([normal[0], normal[1], normal[2], d])

        # Refine with all inliers using least squares
        if best_inliers is not None and best_n_inliers > 3:
            inlier_points = points[best_inliers]
            
            # Fit plane using SVD (minimize perpendicular distances)
            centroid = np.mean(inlier_points, axis=0)
            centered = inlier_points - centroid
            _, _, vh = np.linalg.svd(centered)
            normal = vh[-1]  # Last row of V^T is the normal

            # Ensure normal points up
            if normal[2] < 0:
                normal = -normal

            d = -np.dot(normal, centroid)
            best_plane = np.array([normal[0], normal[1], normal[2], d])

        return best_plane, best_inliers
    

    def plane_to_angles(self, plane: np.ndarray):
        """
        Convert plane normal to pitch and roll angles.
        
        Assumes plane normal [a, b, c] where ideal floor has normal [0, 0, 1].
        
        Returns:
            (pitch_deg, roll_deg) - angles in degrees
            
        Convention (ROS standard, X-forward, Y-left, Z-up):
            - Pitch: rotation about Y-axis (positive = nose up)
            - Roll: rotation about X-axis (positive = right side down)
        """
        a, b, c, _ = plane
        
        # Normal should be unit vector, but normalize just in case
        norm = np.sqrt(a*a + b*b + c*c)
        a, b, c = a/norm, b/norm, c/norm

        # Pitch: angle between normal and vertical in XZ plane
        # tan(pitch) = -a / c (negative because positive pitch tips sensor back)
        pitch_rad = np.arctan2(-a, c)

        # Roll: angle between normal and vertical in YZ plane  
        # tan(roll) = b / c (positive roll tips right side down)
        roll_rad = np.arctan2(b, c)

        return np.rad2deg(pitch_rad), np.rad2deg(roll_rad)
    
    
    def run_estimation(self):
        """Kick off plane estimation in background thread."""
        if self.estimation_running:
            self.get_logger().info('Estimation already running, skipping...')
            return
            
        with self.lock:
            if self.total_points < self.min_points:
                self.get_logger().info(
                    f'Not enough points yet: {self.total_points}/{self.min_points}'
                )
                return
            
            # Grab a snapshot of the points
            all_points = np.vstack(self.accumulated_points).copy()
            n_points = self.total_points
        
        self.get_logger().info(f'Starting RANSAC on {n_points} points in background...')
        self.estimation_running = True
        self.thread_pool.submit(self._run_estimation_thread, all_points)


    def _run_estimation_thread(self, all_points: np.ndarray):
        """Run plane estimation on the provided points snapshot.

        Args:
            all_points: Nx3 array of points (snapshot taken under lock)
        """
        try:
            n_points = len(all_points)
            self.get_logger().info(f'Running RANSAC on {n_points} points...')

            # Fit plane
            plane, inliers = self.fit_plane_ransac(all_points)

            if plane is None:
                self.get_logger().error('Failed to fit plane')
                return

            n_inliers = np.sum(inliers) if inliers is not None else 0
            inlier_ratio = n_inliers / n_points

            # Get angles
            pitch_deg, roll_deg = self.plane_to_angles(plane)

            # Compute statistics on inlier heights
            if inliers is not None:
                inlier_points = all_points[inliers]
                # Height of floor (distance from sensor to floor along Z)
                floor_height = -plane[3] / plane[2] if abs(plane[2]) > 1e-6 else 0
                height_std = np.std(inlier_points[:, 2])
            else:
                floor_height = 0
                height_std = 0

            # Log results
            self.get_logger().info(
                f'\n{"="*50}\n'
                f'FLOOR PLANE ESTIMATION RESULTS\n'
                f'{"="*50}\n'
                f'Plane equation: {plane[0]:.6f}x + {plane[1]:.6f}y + {plane[2]:.6f}z + {plane[3]:.6f} = 0\n'
                f'Normal vector: [{plane[0]:.6f}, {plane[1]:.6f}, {plane[2]:.6f}]\n'
                f'{"="*50}\n'
                f'PITCH: {pitch_deg:+.3f}° (rotation about Y-axis)\n'
                f'ROLL:  {roll_deg:+.3f}° (rotation about X-axis)\n'
                f'{"="*50}\n'
                f'Floor height (Z): {floor_height:.3f} m\n'
                f'Inliers: {n_inliers} / {n_points} ({inlier_ratio*100:.1f}%)\n'
                f'Inlier height std: {height_std*1000:.1f} mm\n'
                f'{"="*50}'
            )

            # Publish results
            msg = Float64MultiArray()
            msg.data = [
                float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]),
                float(pitch_deg), float(roll_deg),
                float(n_inliers),
                float(floor_height),
                float(height_std)
            ]
            self.plane_pub.publish(msg)

            # Publish RViz visualization
            self._publish_visualization(plane, floor_height)

        except Exception as e:
            self.get_logger().error(f'Estimation failed: {e}')
        finally:
            self.estimation_running = False

    def _publish_visualization(self, plane: np.ndarray, floor_height: float):
        """Publish RViz markers for the estimated plane and normal vector.

        Args:
            plane: [a, b, c, d] plane equation coefficients (normal + offset)
            floor_height: Z height of the floor plane
        """
        now = self.get_clock().now().to_msg()
        frame_id = 'utlidar_lidar'  # Match the point cloud frame

        normal = plane[:3]

        # --- Plane marker (semi-transparent rectangle) ---
        plane_marker = Marker()
        plane_marker.header.stamp = now
        plane_marker.header.frame_id = frame_id
        plane_marker.ns = 'floor_plane'
        plane_marker.id = 0
        plane_marker.type = Marker.CUBE
        plane_marker.action = Marker.ADD

        # Position at floor height, centered at origin
        plane_marker.pose.position.x = 0.0
        plane_marker.pose.position.y = 0.0
        plane_marker.pose.position.z = float(floor_height)

        # Orientation: rotate from Z-up to match plane normal
        # Compute quaternion that rotates [0,0,1] to plane normal
        quat = self._rotation_to_align_z_with_normal(normal)
        plane_marker.pose.orientation.x = float(quat[0])
        plane_marker.pose.orientation.y = float(quat[1])
        plane_marker.pose.orientation.z = float(quat[2])
        plane_marker.pose.orientation.w = float(quat[3])

        # Size: 4m x 4m plane, very thin
        plane_marker.scale.x = 4.0
        plane_marker.scale.y = 4.0
        plane_marker.scale.z = 0.01

        # Semi-transparent green
        plane_marker.color.r = 0.2
        plane_marker.color.g = 0.8
        plane_marker.color.b = 0.2
        plane_marker.color.a = 0.5

        plane_marker.lifetime.sec = 0  # Persistent until next update

        self.plane_marker_pub.publish(plane_marker)

        # --- Normal vector marker (arrow) ---
        normal_marker = Marker()
        normal_marker.header.stamp = now
        normal_marker.header.frame_id = frame_id
        normal_marker.ns = 'floor_normal'
        normal_marker.id = 0
        normal_marker.type = Marker.ARROW
        normal_marker.action = Marker.ADD

        # Arrow from floor, extending 2m along normal direction
        arrow_length = 2.0
        start = Point()
        start.x = 0.0
        start.y = 0.0
        start.z = float(floor_height)

        end = Point()
        end.x = float(normal[0] * arrow_length)
        end.y = float(normal[1] * arrow_length)
        end.z = float(floor_height + normal[2] * arrow_length)

        normal_marker.points = [start, end]

        # Arrow dimensions (larger for visibility)
        normal_marker.scale.x = 0.1   # Shaft diameter
        normal_marker.scale.y = 0.2   # Head diameter
        normal_marker.scale.z = 0.0   # Head length (auto)

        # Bright red color
        normal_marker.color.r = 1.0
        normal_marker.color.g = 0.0
        normal_marker.color.b = 0.0
        normal_marker.color.a = 1.0

        normal_marker.lifetime.sec = 0

        self.normal_marker_pub.publish(normal_marker)
        self.get_logger().info(f'Published markers at floor_height={floor_height:.2f}m')

    def _rotation_to_align_z_with_normal(self, normal: np.ndarray) -> np.ndarray:
        """Compute quaternion that rotates [0,0,1] to the given normal.

        Args:
            normal: Unit normal vector [a, b, c]

        Returns:
            Quaternion [x, y, z, w]
        """
        z_axis = np.array([0.0, 0.0, 1.0])
        normal = np.array(normal, dtype=np.float64)

        dot = np.dot(z_axis, normal)

        if dot > 0.9999:
            # Already aligned
            return np.array([0.0, 0.0, 0.0, 1.0])
        elif dot < -0.9999:
            # Opposite direction - rotate 180 degrees around X
            return np.array([1.0, 0.0, 0.0, 0.0])

        # Rotation axis = cross(z_axis, normal)
        axis = np.cross(z_axis, normal)
        axis = axis / np.linalg.norm(axis)

        # Rotation angle
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        # Quaternion from axis-angle
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        return np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half
        ])

    def reset_accumulator(self):
        """Clear accumulated points."""
        with self.lock:
            self.accumulated_points = []
            self.total_points = 0
            self.clouds_received = 0
        self.get_logger().info('Accumulator reset')


def main(args=None):
    """
    The main function.
    :param args: Not used directly by the user, but used by ROS2 to configure
    certain aspects of the Node.
    """
    rclpy.init(args=args)
    node = PCTransform()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Run final estimation before shutting down
        node.get_logger().info('Shutting down, running final estimation...')
        node.run_estimation()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()