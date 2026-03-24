#! usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math

# from shell_car_model import shell_car_model


class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(PointCloud2, '/scan/points', self.listener_callback, 10)
        self.get_logger().info("Lidar Subscriber Node has been started")

    def listener_callback(self, msg):
        self.get_logger().info(f'Received LIDAR data with {msg.width} points')
        
        try:
            # Read points from the point cloud
            points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            
            # Convert generator to list
            points_list = list(points)
            
            if not points_list:
                self.get_logger().warn("No valid points in LIDAR data")
                return
            
            # Filter out points that are too close (likely sensor noise/minimum range)
            # AND filter out ground points AND filter robot body
            filtered_points = []
            for x, y, z in points_list:
                distance = (x**2 + y**2 + z**2)**0.5
                
                # Multi-stage filtering:
                # 1. Minimum distance (ignore very close points - sensor noise)
                # 2. Ground filtering (ignore floor - but keep low obstacles like curbs)
                # 3. Robot body filtering (ignore points too close in XY plane)
                xy_distance = (x**2 + y**2)**0.5  # Horizontal distance only
                
                if distance > 0.3 and z > -0.10 and xy_distance > 0.25:
                    filtered_points.append((x, y, z))
            
            if not filtered_points:
                self.get_logger().warn("All points filtered out (too close to sensor or ground points)")
                return
            
            points_list = filtered_points
            
            # Calculate distances for all points
            distances = [(x**2 + y**2 + z**2)**0.5 for x, y, z in points_list]
            closest = min(distances)
            
            self.get_logger().info(f"Closest 3D point: {closest:.2f}m")
            
            # Separate points by direction using angle-based approach
            # Front: -45° to +45° (ahead)
            # Right: -90° to -45° (right side)
            # Back: 90° to 180° or -180° to -90° (behind)
            # Left: 45° to 90° (left side)
            
            front_points = []
            back_points = []
            right_points = []
            left_points = []
            
            for x, y, z in points_list:
                angle = math.degrees(math.atan2(y, x))
                
                # Front: -45 to +45 degrees
                if -45 <= angle <= 45:
                    front_points.append((x, y, z))
                # Right: -90 to -45 degrees
                elif -90 <= angle < -45:
                    right_points.append((x, y, z))
                # Left: 45 to 90 degrees
                elif 45 < angle <= 90:
                    left_points.append((x, y, z))
                # Back: 90 to 180 or -180 to -90 degrees
                else:
                    back_points.append((x, y, z))
            
            # Function to analyze a set of points
            def analyze_direction(points, direction_name):
                if not points:
                    return None
                
                try:
                    distances = [(x**2 + y**2 + z**2)**0.5 for x, y, z in points]
                    closest = min(distances)
                    avg = sum(distances) / len(distances)
                    count = len(points)
                    return {
                        'name': direction_name,
                        'closest': closest,
                        'average': avg,
                        'count': count
                    }
                except (ZeroDivisionError, ValueError):
                    self.get_logger().error(f"Error in {direction_name}")
                    return None
            
            # Analyze all directions
            directions = [
                analyze_direction(front_points, "Front"),
                analyze_direction(back_points, "Back"),
                analyze_direction(left_points, "Left"),
                analyze_direction(right_points, "Right")
            ]
            
            # Log results for all directions
            for direction in directions:
                if direction:
                    self.get_logger().info(
                        f"{direction['name']}: {direction['count']} points, "
                        f"Closest: {direction['closest']:.2f}m, "
                        f"Avg: {direction['average']:.2f}m"
                    )
            
            # Overall statistics
            all_distances = [(x**2 + y**2 + z**2)**0.5 for x, y, z in points_list]
            
            if not all_distances:
                self.get_logger().warn("No valid distances calculated")
                return
                
            overall_closest = min(all_distances)
            valid_distances = [d for d in all_distances if d > 0 and d != float('inf')]
            overall_avg = sum(valid_distances) / len(valid_distances) if valid_distances else 0
            
            self.get_logger().info(f"OVERALL - Closest: {overall_closest:.2f}m, Avg: {overall_avg:.2f}m")
            
            # DEBUG: Show coordinates of closest point
            closest_point = min(points_list, key=lambda p: (p[0]**2 + p[1]**2 + p[2]**2)**0.5)
            self.get_logger().info(f"DEBUG: Closest point - X: {closest_point[0]:.3f}m, Y: {closest_point[1]:.3f}m, Z: {closest_point[2]:.3f}m")
            
            # Check for obstacles - only warn if REALLY close
            obstacle_threshold = 0.34  # 60cm threshold
            
            if overall_closest < obstacle_threshold:
                self.get_logger().warn(f"OBSTACLE DETECTED at {overall_closest:.2f}m!")
                
                # Find which direction the obstacle is in using tolerance
                tolerance = 0.05 # 5cm tolerance for direction classification
                
                obstacle_directions = []
                
                # Check each direction
                if front_points:
                    front_closest = min([(x**2 + y**2 + z**2)**0.5 for x, y, z in front_points])
                    if abs(front_closest - overall_closest) < tolerance:
                        obstacle_directions.append(("FRONT", front_closest))
                
                if back_points:
                    back_closest = min([(x**2 + y**2 + z**2)**0.5 for x, y, z in back_points])
                    if abs(back_closest - overall_closest) < tolerance:
                        obstacle_directions.append(("BACK", back_closest))
                
                if left_points:
                    left_closest = min([(x**2 + y**2 + z**2)**0.5 for x, y, z in left_points])
                    if abs(left_closest - overall_closest) < tolerance:
                        obstacle_directions.append(("LEFT", left_closest))
                
                if right_points:
                    right_closest = min([(x**2 + y**2 + z**2)**0.5 for x, y, z in right_points])
                    if abs(right_closest - overall_closest) < tolerance:
                        obstacle_directions.append(("RIGHT", right_closest))
                
                if obstacle_directions:
                    # Find the closest direction
                    closest_dir = min(obstacle_directions, key=lambda x: x[1])
                    self.get_logger().warn(f"Obstacle is in {closest_dir[0]}!")
                else:
                    self.get_logger().warn("Obstacle direction could not be determined!")
            else:
                self.get_logger().info("Path clear in all directions!")
                
        except Exception as e:
            self.get_logger().error(f"Error processing LIDAR data: {e}")


def main(args=None):
    rclpy.init(args=args)
    lidar_subscriber = LidarSubscriber()
    rclpy.spin(lidar_subscriber)
    lidar_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
