import os
import sys
import argparse
import time
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.backends import cudnn
import random

from new_models.ultraseg108k import UltraSeg108
from new_models.ultraseg130k import UltraSeg130


import torch
import os
import torch
print(f"PyTorch 实际使用的线程数: {torch.get_num_threads()}")  # 应该显示 16
# # 查看 PyTorch 默认使用的线程数（通常等于你的 CPU 核心数）
# print(f"PyTorch 使用的 CPU 线程数: {torch.get_num_threads()}")
# print(f"PyTorch 使用的 CPU 核心数 (inter-op): {torch.get_num_interop_threads()}")

# 如果你想限制只用 2 个核心（避免电脑卡死），取消下面注释：
# torch.set_num_threads(2)

# 查看系统总核心数（Python 方式）
import multiprocessing
print(f"系统总 CPU 核心数: {multiprocessing.cpu_count()}")
class RealTimePolypSegmentor:
    def __init__(self, model_path, algo='ultraseg130', device='cpu', input_size=256):
        """
        Initialize real-time polyp segmentation pipeline.

        Args:
            model_path: Path to model checkpoint (best.pth)
            algo: Model architecture ('ultraseg130' or 'ultraseg108')
            device: Computation device ('cpu' or 'cuda')
            input_size: Input resolution (default: 256)
        """
        self.device = torch.device(device)
        self.algo = algo
        self.input_size = input_size
        self.num_classes = 2  # Background and Polyp

        # Set random seed for reproducibility
        self._setup_random_seed(42)

        # Load model
        self.model = self._initialize_model(model_path)
        self.model.eval()

        # FPS tracking
        self.fps_history = deque(maxlen=10)  # Smooth over last 10 frames
        self.prev_time = time.time()

        # Visualization settings
        self.mask_color = np.array([0, 0, 255], dtype=np.uint8)  # Red in BGR
        self.contour_color = (0, 255, 0)  # Green contours
        self.alpha = 0.5  # Overlay transparency

        print(f"[INFO] Model loaded successfully: {algo}")
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Input resolution: {input_size}x{input_size}")
        print(f"[INFO] Model parameters: ~130K")

        # Warm-up to avoid first-frame lag
        self._warmup_model()

    def _setup_random_seed(self, seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def _initialize_model(self, model_path):
        """Initialize model architecture and load weights."""
        if self.algo == "ultraseg108":
            model = UltraSeg108(in_ch=3, out_ch=self.num_classes)
        elif self.algo == "ultraseg130":
            model = UltraSeg130(in_ch=3, out_ch=self.num_classes, key=3)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algo}")

        model = model.to(self.device)

        # Load checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        print(f"[INFO] Checkpoint loaded: {model_path}")

        return model

    def _warmup_model(self):
        """Run dummy inference to warm up the model (avoid cold start)."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        with torch.no_grad():
            _, _, _ = self.model(dummy_input)
        print("[INFO] Model warmup completed")

    def preprocess(self, frame):
        """
        Preprocess video frame for inference.

        Args:
            frame: OpenCV BGR image (H, W, 3)

        Returns:
            tensor: Preprocessed tensor (1, 3, 256, 256)
            resized: Resized BGR frame for visualization
        """
        # Resize to model input size (256x256)
        resized = cv2.resize(frame, (self.input_size, self.input_size))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Convert to tensor (HWC -> CHW) and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device), resized

    def postprocess(self, logits, original_frame):
        """
        Post-process model predictions.

        Args:
            logits: Model output logits (1, 2, 256, 256)
            original_frame: Resized BGR frame (256, 256, 3)

        Returns:
            vis_frame: Visualization with mask overlay
            binary_mask: Binary segmentation mask
        """
        # Get predicted class (argmax)
        pred_mask = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Create colored mask (red for polyp)
        colored_mask = np.zeros_like(original_frame)
        colored_mask[pred_mask == 1] = self.mask_color  # Class 1: Polyp

        # Overlay mask on original frame
        vis_frame = cv2.addWeighted(original_frame, 1.0, colored_mask, self.alpha, 0)

        # Find and draw contours for better boundary visualization
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_frame, contours, -1, self.contour_color, 2)

        return vis_frame, pred_mask

    def calculate_fps(self, inference_time_ms):
        """Calculate smoothed FPS based on inference time."""
        current_time = time.time()
        instant_fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_history.append(instant_fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        return avg_fps, instant_fps

    def draw_info_panel(self, frame, fps, inference_time):
        """
        Draw FPS, inference time, and total frame time on top-right corner.
        """
        # 准备文本（3行信息）
        infer_text = f"Infer FPS: {inference_time:.1f}ms"
        total_text = f"Frame: {1000 / fps:.1f}ms" if fps > 0 else "Frame: N/A"
        model_text = f"Model: {self.algo}"

        # 字体设置
        infer_scale = 0.4
        total_scale = 0.4
        model_scale = 0.35

        # 获取文本尺寸（用于右对齐）
        infer_size = cv2.getTextSize(infer_text, cv2.FONT_HERSHEY_SIMPLEX, infer_scale, 2)[0]
        total_size = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, total_scale, 2)[0]
        model_size = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, model_scale, 2)[0]

        # 右对齐计算
        x_infer = frame.shape[1] - infer_size[0] - 10
        x_total = frame.shape[1] - total_size[0] - 10
        x_model = frame.shape[1] - model_size[0] - 10

        # 背景框
        overlay = frame.copy()
        box_height = 55  # 稍微减小高度，3行文字不需要60
        box_width = 140
        cv2.rectangle(overlay,
                      (frame.shape[1] - box_width, 5),
                      (frame.shape[1] - 5, box_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 绘制文字（Y坐标往上提：20, 38, 56）
        # 注意：基线(bottom-left origin)，所以y=20表示文字底部在y=20
        cv2.putText(frame, infer_text, (x_infer, 20), cv2.FONT_HERSHEY_SIMPLEX, infer_scale, (0, 255, 255), 2)
        cv2.putText(frame, total_text, (x_total, 38), cv2.FONT_HERSHEY_SIMPLEX, total_scale, (255, 255, 255), 2)
        cv2.putText(frame, model_text, (x_model, 56), cv2.FONT_HERSHEY_SIMPLEX, model_scale, (200, 200, 200), 1)

    def process_frame(self, frame):
        """
        Process a single frame through the segmentation pipeline.

        Returns:
            vis_result: Visualized result
            inference_time: Time taken for inference (ms)
        """
        # Preprocess
        input_tensor, resized_frame = self.preprocess(frame)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            _, _, logits = self.model(input_tensor)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Postprocess
        vis_result, mask = self.postprocess(logits, resized_frame)

        # Calculate and draw FPS
        fps, _ = self.calculate_fps(inference_time)
        self.draw_info_panel(vis_result, fps, inference_time)

        return vis_result, inference_time, mask

    def run(self, video_path, output_path=None, display=True):
        """
        Run real-time segmentation on video source.

        Args:
            video_path: Path to video file or camera index ('0', '1')
            output_path: Optional path to save output video
            display: Whether to show real-time display window
        """
        # Open video source
        if video_path.isdigit():
            source = int(video_path)
            print(f"[INFO] Opening camera device: {source}")
        else:
            source = video_path
            if not os.path.exists(source):
                raise FileNotFoundError(f"Video file not found: {source}")
            print(f"[INFO] Opening video file: {source}")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not isinstance(source, int) else -1

        print(f"[INFO] Video FPS: {fps:.2f}")
        if total_frames > 0:
            print(f"[INFO] Total frames: {total_frames}")

        # Initialize video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (self.input_size, self.input_size))
            print(f"[INFO] Saving output to: {output_path}")

        frame_count = 0
        total_inference_time = 0

        print("[INFO] Starting inference loop...")
        print("[CONTROLS] Press 'Q' to quit | Press 'S' to save screenshot")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream")
                    break

                # Process frame
                vis_result, infer_time, mask = self.process_frame(frame)
                total_inference_time += infer_time
                frame_count += 1

                # Save to video file
                if writer:
                    writer.write(vis_result)

                # Display results
                if display:
                    # Add progress bar for video files
                    if total_frames > 0:
                        progress = (frame_count / total_frames) * 100
                        # cv2.putText(vis_result, f"Progress: {progress:.1f}%",
                        #             (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow("Real-time Polyp Segmentation (256x256)", vis_result)

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Interrupted by user")
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_frame{frame_count}.png"
                        cv2.imwrite(screenshot_path, vis_result)
                        print(f"[INFO] Screenshot saved: {screenshot_path}")

                # Print statistics every 30 frames
                if frame_count % 30 == 0:
                    avg_infer = total_inference_time / frame_count
                    current_fps = 1000.0 / avg_infer if avg_infer > 0 else 0
                    print(f"[STATS] Frames: {frame_count} | Avg Inference: {avg_infer:.2f}ms | FPS: {current_fps:.1f}")

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            # Final statistics
            if frame_count > 0:
                avg_time = total_inference_time / frame_count
                print(f"\n[INFO] Processing complete:")
                print(f"  - Total frames processed: {frame_count}")
                print(f"  - Average inference time: {avg_time:.2f} ms")
                print(f"  - Average FPS: {1000.0 / avg_time:.1f}")
                print(f"  - Real-time capability: {'YES' if avg_time < 33 else 'NO'} (target: <33ms for 30fps)")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Polyp Segmentation for Colonoscopy Videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file with CPU:
  python predict_video.py --model best.pth --source video.avi --device cpu

  # Use webcam (device 0):
  python predict_video.py --model best.pth --source 0 --device cpu

  # Save results to file:
  python predict_video.py --model best.pth --source video.avi --output result.avi
        """
    )

    parser.add_argument('--model', type=str, default=r'G:\Github-folder\UltraSeg\ultraseg\best.pth',
                        help='Path to model checkpoint (e.g., best.pth)')
    parser.add_argument('--algo', type=str, default='ultraseg130',
                        choices=['ultraseg130', 'ultraseg108'],
                        help='Model architecture (default: ultraseg130)')
    parser.add_argument('--source', type=str, default=r'G:\Github-folder\UltraSeg\ultraseg\video\4.avi',
                        help='Video source: "0" for webcam, or path to video file')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Computation device (default: cpu)')
    parser.add_argument('--output', type=str, default="result4.avi",
                        help='Optional: Path to save output video')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without display window (useful for batch processing)')

    args = parser.parse_args()

    # Initialize segmentor
    segmentor = RealTimePolypSegmentor(
        model_path=args.model,
        algo=args.algo,
        device=args.device,
        input_size=256
    )

    # Run segmentation
    segmentor.run(
        video_path=args.source,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()