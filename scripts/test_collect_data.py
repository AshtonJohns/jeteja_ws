# ROS2 RealSense, LiDAR, and Joystick Data Collection

import os
import sys
import json
import csv
from time import time, sleep
from datetime import datetime
import pygame
import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Load configs
params_file_path = os.path.join(sys.path[0], 'test_config.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
STEERING_CENTER = params['steering_center']
STEERING_RANGE = params['steering_range']
THROTTLE_AXIS = params['throttle_joy_axis']
THROTTLE_STALL = params['throttle_stall']
THROTTLE_FWD_RANGE = params['throttle_fwd_range']
THROTTLE_REV_RANGE = params['throttle_rev_range']
THROTTLE_LIMIT = params['throttle_limit']
THROTTLE_MIN = params['throttle_min']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']

# LiDAR Node for ROS2
class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.lidar_data = []
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

    def lidar_callback(self, msg):
        # Store the ranges (distance readings)
        self.lidar_data = msg.ranges

# Initialize hardware (serial communication and joystick)
try:
    ser_pico = serial.Serial(port='/dev/ttyACM1', baudrate=115200)
except:
    ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)

# Initialize joystick and directories for saving data
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
rgb_image_dir = os.path.join(data_dir, 'rgb_images/')
lidar_image_dir = os.path.join(data_dir, 'lidar_images/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(rgb_image_dir, exist_ok=True)
os.makedirs(lidar_image_dir, exist_ok=True)

# Initialize RealSense camera pipeline for RGB
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)

# Start streaming from the camera
pipeline.start(config)

# Initialize variables
is_recording = False
frame_counts = 0
ax_val_st = 0.
ax_val_th = 0.

# Initialize Pygame for joystick handling
pygame.init()

# Initialize ROS2 LiDAR Node
rclpy.init()  # Initialize ROS2
lidar_node = LidarNode()  # Create the LiDAR node

# Write CSV headers if the file doesn't exist
if not os.path.exists(label_path):
    with open(label_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "steering", "throttle", "lidar_ranges"])

try:
    while True:
        # Capture RGB frames from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue  # Skip if frame is not ready

        # Convert RGB to numpy array and resize to 120x160
        color_image = np.asanyarray(color_frame.get_data())
        resized_color_image = cv2.resize(color_image, (160, 120))

        # Get the LiDAR data
        lidar_data = np.array(lidar_node.lidar_data)

        # Replace 'inf' values with 25 meters (max range of A3 LiDAR) before padding
        lidar_data[np.isinf(lidar_data)] = 25.0

        # Pad the LiDAR data to match the required size of 120x160
        lidar_data = np.pad(lidar_data, (0, 160 * 120 - len(lidar_data)), constant_values=25)

        # Reshape to match the required image size (120x160)
        lidar_image = lidar_data.reshape(120, 160)

        # Ensure there are no additional issues with reshaping
        if lidar_image.shape != (120, 160):
            print("Error: LiDAR data cannot be reshaped to (120, 160)")
            continue

        # Display the RGB image for visualization
        cv2.imshow('RGB Stream', resized_color_image)

        # Handle joystick inputs
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    print("Collecting data")
                    is_recording = not is_recording  # Toggle recording
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    msg = ("END,END\n").encode('utf-8')
                    ser_pico.write(msg)
                    raise KeyboardInterrupt

        # Calculate steering and throttle values
        act_st = -ax_val_st
        act_th = -ax_val_th
        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))

        # Refined throttle control with correct forward and reverse mapping
        if act_th > 0:
            # Forward motion with variable speed control
            duty_th = THROTTLE_STALL + int((THROTTLE_FWD_RANGE - THROTTLE_STALL) * act_th)
        elif act_th < 0:
            # Reverse motion with variable speed control
            duty_th = THROTTLE_STALL - int((THROTTLE_STALL - THROTTLE_REV_RANGE) * abs(act_th))
        else:
            # No throttle
            duty_th = THROTTLE_STALL

        # Send control signals to the microcontroller
        ser_pico.write((f"{duty_st},{duty_th}\n").encode('utf-8'))

        # Spin the LiDAR Node to process ROS callbacks
        rclpy.spin_once(lidar_node)

        # Get the latest LiDAR data
        lidar_ranges = lidar_node.lidar_data

        # Save data if recording is active
        if is_recording:
            # Save RGB and LiDAR images
            rgb_image_name = f"{frame_counts}_rgb.png"
            lidar_image_name = f"{frame_counts}_lidar.png"

            cv2.imwrite(os.path.join(rgb_image_dir, rgb_image_name), resized_color_image)
            cv2.imwrite(os.path.join(lidar_image_dir, lidar_image_name), lidar_image)

            # Log joystick values and LiDAR ranges with image name
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([lidar_image_name, ax_val_st, ax_val_th, lidar_ranges])

            frame_counts += 1  # Increment frame counter

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Terminated by user.")

finally:
    # Cleanup
    pipeline.stop()
    pygame.quit()
    ser_pico.close()
    rclpy.shutdown()
