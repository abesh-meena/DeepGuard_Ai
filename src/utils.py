"""
src/utils.py

Helper utilities for loading images/videos and basic I/O.
"""
import os, io
import numpy as np
from PIL import Image

# Try to import cv2 for video loading
try:
    import cv2
except Exception as e:
    cv2 = None

def load_image_as_array(path, target_size=(224,224)):
    """
    Load image file and return numpy array resized to target_size.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.uint8)
    return arr

def save_numpy_as_image(arr, out_path):
    img = Image.fromarray(arr.astype('uint8'))
    img.save(out_path)
    return out_path

def load_video_as_frames(path, max_frames=20, target_size=(224, 224)):
    """
    Load video file and return frames as numpy array.
    
    Args:
        path: Path to video file
        max_frames: Maximum number of frames to extract (0 = all frames)
        target_size: Target size for frames (default: (224, 224))
    
    Returns:
        numpy array of frames with shape (num_frames, H, W, 3)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available. Install opencv-python to use load_video_as_frames.")
    
    from src.model import load_video
    return load_video(path, max_frames=max_frames, resize=target_size)

def is_video_file(file_path):
    """
    Check if file is a video file based on extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if file is a video, False otherwise
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)

def is_image_file(file_path):
    """
    Check if file is an image file based on extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if file is an image, False otherwise
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)
