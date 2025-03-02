import os
import sys
import json
from test_hardware import get_realsense_frame, setup_realsense_camera, setup_serial, setup_joystick, encode_dutycylce, encode
import pygame
import cv2 as cv
from time import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Adds dummy to run Pygame without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Pygame modules
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

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
PAUSE_BUTTON = params['pause_btn']
STOP_BUTTON = params['stop_btn']

# Initialize hardware
try:
    ser_pico = setup_serial(port='/dev/ttyACM0', baudrate=115200)
except:
    ser_pico = setup_serial(port='/dev/ttyACM1', baudrate=115200)
cam = setup_realsense_camera()
js = setup_joystick()
is_paused = True

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_path = "/home/ucajetson/UCAJetson/models/TensorRT_1.trt"  # Path to the TensorRT model

with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

# Manual safe type mapping
def safe_trt_nptype(trt_dtype):
    """Manual dtype mapping to avoid deprecated np.bool and other issues."""
    if trt_dtype == trt.bool:
        return np.bool_
    elif trt_dtype == trt.int8:
        return np.int8
    elif trt_dtype == trt.int32:
        return np.int32
    elif trt_dtype == trt.float16:
        return np.float16
    elif trt_dtype == trt.float32:
        return np.float32
    else:
        raise TypeError(f"Unsupported dtype encountered: {trt_dtype}")

# Allocate buffers for TensorRT inference
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding))
        dtype = safe_trt_nptype(engine.get_tensor_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

inputs, outputs, bindings, stream = allocate_buffers(engine)

# Frame rate calculation variables
prev_time = time()
frame_count = 0
fps = 0

# MAIN LOOP
try:
    while True:
        ret, color_image = get_realsense_frame(cam)  # Capture RGB frame
        if not ret or color_image is None:
            print("No frame received. TERMINATE!")
            break

        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(PAUSE_BUTTON):
                    is_paused = not is_paused
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    ser_pico.write(b"END,END\n")
                    raise KeyboardInterrupt

        # Resize and normalize RGB image
        color_image_resized = cv.resize(color_image, (160, 120))
        color_image_normalized = color_image_resized.astype(np.float32) / 255.0

        # Prepare the tensor for TensorRT
        img_tensor = color_image_normalized.transpose(2, 0, 1)  # Shape (3, 120, 160)
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension, now shape is (1, 3, 120, 160)

        # Copy img_tensor to TensorRT input buffer
        np.copyto(inputs[0][0], img_tensor.ravel())

        # Run inference with TensorRT
        cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
        stream.synchronize()

        # Retrieve and process predictions
        pred_st, pred_th = outputs[0][0][:2]
        st_trim = max(min(float(pred_st), 0.999), -0.999)
        th_trim = max(min(float(pred_th), 0.999), -0.999)

        # Encode and send commands
        if not is_paused:
            msg = encode_dutycylce(st_trim, th_trim, params)
        else:
            duty_st, duty_th = params['steering_center'], params['throttle_stall']
            msg = encode(duty_st, duty_th)

        ser_pico.write(msg)

        # Calculate and print frame rate
        frame_count += 1
        current_time = time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            print(f"Autopilot Frame Rate: {fps:.2f} FPS")
            prev_time = current_time
            frame_count = 0

except KeyboardInterrupt:
    print("Terminated by user.")
finally:
    pygame.joystick.quit()
    ser_pico.close()
    cv.destroyAllWindows()
