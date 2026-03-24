import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import pygame
import sys

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('teleop_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Initialize pygame properly
        pygame.init()
        pygame.display.set_mode((400, 300))
        pygame.display.set_caption("ROS2 Teleop - Use Arrow Keys, Q/E for Curves")
        
        self.linear_speed = 15.0
        self.angular_speed = 3.0
        self.get_logger().info("Teleop node started - Use arrow keys to control, Q/E for curved motion, ESC to quit")

    def run(self):
        self.get_logger().info("Running main loop... Focus on the pygame window and use keys")
        running = True
        clock = pygame.time.Clock()
        
        # List of keys that, when released, should stop the robot
        self.keys_to_stop = [pygame.K_UP, pygame.K_DOWN, pygame.K_q, pygame.K_e]
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    twist = Twist()
                    
                    if event.key == pygame.K_UP:
                        twist.linear.x = self.linear_speed
                        self.get_logger().info("Moving forward (straight)")
                    elif event.key == pygame.K_DOWN:
                        twist.linear.x = -self.linear_speed
                        self.get_logger().info("Moving backward (straight)")
                    elif event.key == pygame.K_q:
                        # Q key: Forward linear speed + Positive angular speed (Left turn)
                        twist.linear.x = self.linear_speed
                        twist.angular.z = self.angular_speed
                        self.get_logger().info("Moving forward and turning left (Curve)")
                    elif event.key == pygame.K_e:
                        # E key: Forward linear speed + Negative angular speed (Right turn)
                        twist.linear.x = self.linear_speed
                        twist.angular.z = -self.angular_speed
                        self.get_logger().info("Moving forward and turning right (Curve)")
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        continue
                    else:
                        # Skip publishing if a non-movement key is pressed
                        continue 
                    
                    self.publisher_.publish(twist)
                
                elif event.type == pygame.KEYUP:
                    # Check if one of the movement keys was released
                    if event.key in self.keys_to_stop:
                        twist = Twist()  # All values default to 0
                        self.publisher_.publish(twist)
                        self.get_logger().info("Stopped")
            
            # Update Pygame display (rest of the code is unchanged)
            screen = pygame.display.get_surface()
            screen.fill((50, 50, 50))
            
            font = pygame.font.Font(None, 36)
            text = font.render("ROS2 Teleop Active", True, (255, 255, 255))
            screen.blit(text, (50, 100))
            
            instructions = [
                "UP/DOWN: Straight Move",
                "Q: Forward Left Curve", # Updated
                "E: Forward Right Curve", # Updated
                "ESC: Quit",
                "Make sure this window has focus!"
            ]
            
            small_font = pygame.font.Font(None, 24)
            for i, instruction in enumerate(instructions):
                text = small_font.render(instruction, True, (255, 255, 255))
                screen.blit(text, (50, 180 + i * 30)) # Moved text down a bit
            
            pygame.display.flip()
            clock.tick(30)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Publish a final stop command before shutting down
        final_stop = Twist()
        node.publisher_.publish(final_stop)
        
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()