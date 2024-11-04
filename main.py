import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Load the video
video_path = "ENTER_VIDEO_NAME_HERE.mp4"  # Ensure correct path and extension
cap = cv2.VideoCapture(video_path)

# ASCII characters used for mapping (from dark to light)
ascii_chars = "@%#*+=-:. "

# Output directory for frames
output_dir = "ascii_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

# ASCII frame settings
output_width = 100  # Adjust width for ASCII art scaling
font = ImageFont.load_default()  # Use default bitmap font to avoid font path issues

def pixel_to_ascii(pixel):
    """Convert a pixel to an ASCII character based on brightness."""
    brightness = np.mean(pixel) / 255  # Normalize to range [0, 1]
    return ascii_chars[int(brightness * (len(ascii_chars) - 1))]

def frame_to_ascii(frame):
    """Convert a frame to ASCII art."""
    # Convert frame to grayscale and resize
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_frame.shape
    aspect_ratio = height / width
    new_height = int(output_width * aspect_ratio * 0.5)  # Adjust aspect ratio
    resized_frame = cv2.resize(gray_frame, (output_width, new_height))

    # Create a blank image for ASCII frame
    ascii_image = Image.new("RGB", (output_width * 6, new_height * 10), "black")
    draw = ImageDraw.Draw(ascii_image)

    # Draw ASCII characters on the image
    for y in range(new_height):
        for x in range(output_width):
            char = pixel_to_ascii(resized_frame[y, x])
            draw.text((x * 5, y * 7), char, font=font, fill="white")  # Adjust spacing as needed

    return ascii_image

# Prepare to write video
output_video_path = "ascii_video.mp4"
frame_rate = 24  # Set desired frame rate
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video_writer = None

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to ASCII and save as image
    ascii_frame = frame_to_ascii(frame)
    ascii_frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    ascii_frame.save(ascii_frame_path)

    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(ascii_frame), cv2.COLOR_RGB2BGR)

    # Initialize video writer with the first frame's dimensions
    if video_writer is None:
        height, width, _ = opencv_image.shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write the frame to the video
    video_writer.write(opencv_image)
    frame_count += 1

cap.release()
if video_writer:
    video_writer.release()

print("ASCII video created:", output_video_path)
