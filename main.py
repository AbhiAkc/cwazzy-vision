"""
CWAZZY VISION - Professional Face Detection System
With JSON output, confidence scores, and file support
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import argparse
import time


class CWazzyVision:
    """Professional face detection system"""
    
    def __init__(self):
        """Initialize the system"""
        print("🚀 Starting CWAZZY VISION...")
        
        # Create folders
        Path("data/captured_faces").mkdir(parents=True, exist_ok=True)
        Path("data/output").mkdir(parents=True, exist_ok=True)
        
        # Load Haar Cascade
        print("📦 Loading Haar Cascade...")
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.cascade.empty():
            print("❌ Could not load Haar Cascade!")
            raise Exception("Haar Cascade failed to load")
        
        print("✅ Haar Cascade loaded!")
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_faces': 0,
            'total_time': 0,
            'detections': []
        }
    
    def detect_faces_with_confidence(self, frame):
        """Detect faces and return with pseudo-confidence scores"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect with Haar Cascade
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Calculate confidence scores based on face size and position
        faces_with_confidence = []
        frame_h, frame_w = frame.shape[:2]
        
        for (x, y, w, h) in faces:
            # Calculate confidence (0.0 to 1.0)
            # Larger faces = higher confidence
            # Faces more centered = higher confidence
            size_confidence = min(1.0, (w * h) / (frame_w * frame_h) * 10)
            
            # Center position
            center_x = x + w // 2
            center_y = y + h // 2
            dist_from_center = np.sqrt(
                ((center_x - frame_w/2) ** 2 + (center_y - frame_h/2) ** 2) / 
                ((frame_w/2) ** 2 + (frame_h/2) ** 2)
            )
            center_confidence = 1.0 - (dist_from_center * 0.3)
            
            # Final confidence
            confidence = (size_confidence + center_confidence) / 2
            confidence = max(0.5, min(0.95, confidence))  # Clamp between 0.5-0.95
            
            faces_with_confidence.append({
                'bbox': (x, y, w, h),
                'confidence': round(confidence, 4)
            })
        
        return faces_with_confidence
    
    def draw_detections(self, frame, faces):
        """Draw boxes and confidence scores"""
        for detection in faces:
            x, y, w, h = detection['bbox']
            conf = detection['confidence']
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw confidence
            text = f"Conf: {conf:.2f}"
            cv2.putText(frame, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return frame
    
    def draw_info(self, frame, fps, face_count):
        """Draw FPS and info"""
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {face_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def save_results_json(self, detections, frame_number, timestamp):
        """Save detections as JSON"""
        result = {
            'frame': frame_number,
            'timestamp': timestamp,
            'num_faces': len(detections),
            'detections': [
                {
                    'bbox': {
                        'x': int(det['bbox'][0]),
                        'y': int(det['bbox'][1]),
                        'width': int(det['bbox'][2]),
                        'height': int(det['bbox'][3])
                    },
                    'confidence': det['confidence']
                }
                for det in detections
            ]
        }
        return result
    
    def process_webcam(self, output_json=None):
        """Process webcam stream"""
        print("📹 Starting webcam... Press Q to quit, S to save face")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_times = []
        frame_count = 0
        all_detections = []
        
        while True:
            ret, frame = self.cap_read(cap)
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            start_time = time.time()
            
            # Detect faces
            detections = self.detect_faces_with_confidence(frame)
            
            # Calculate FPS
            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
            
            # Draw
            frame = self.draw_detections(frame, detections)
            frame = self.draw_info(frame, fps, len(detections))
            
            # Store detection
            detection_record = self.save_results_json(detections, frame_count, datetime.now().isoformat())
            all_detections.append(detection_record)
            
            # Display
            cv2.imshow("CWAZZY VISION", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if len(detections) > 0:
                    for idx, det in enumerate(detections):
                        x, y, w, h = det['bbox']
                        face_roi = frame[y:y+h, x:x+w]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename = f"face_{timestamp}_conf{det['confidence']:.2f}.jpg"
                        cv2.imwrite(f"data/captured_faces/{filename}", face_roi)
                    print(f"💾 Saved {len(detections)} face(s)!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save JSON results if requested
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(all_detections, f, indent=2)
            print(f"✅ Results saved to {output_json}")
        
        print(f"✅ Processed {frame_count} frames")
        return all_detections
    
    def process_image(self, image_path, output_json=None):
        """Process single image"""
        print(f"📸 Processing image: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Detect
        detections = self.detect_faces_with_confidence(frame)
        
        # Draw
        frame = self.draw_detections(frame, detections)
        
        # Save annotated image
        output_path = f"data/output/{Path(image_path).stem}_detected.jpg"
        cv2.imwrite(output_path, frame)
        print(f"✅ Saved annotated image to {output_path}")
        
        # Save JSON
        detection_record = self.save_results_json(detections, 0, datetime.now().isoformat())
        if output_json:
            with open(output_json, 'w') as f:
                json.dump([detection_record], f, indent=2)
            print(f"✅ Results saved to {output_json}")
        
        return detections
    
    def process_video(self, video_path, output_json=None, output_video=None):
        """Process video file"""
        print(f"🎬 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer if output specified
        out = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_count = 0
        all_detections = []
        frame_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect
            detections = self.detect_faces_with_confidence(frame)
            
            # Calculate FPS
            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            fps_live = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
            
            # Draw
            frame = self.draw_detections(frame, detections)
            frame = self.draw_info(frame, fps_live, len(detections))
            
            # Store detection
            detection_record = self.save_results_json(detections, frame_count, datetime.now().isoformat())
            all_detections.append(detection_record)
            
            # Write frame
            if out:
                out.write(frame)
            
            # Progress
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        if out:
            out.release()
        
        # Save JSON
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(all_detections, f, indent=2)
            print(f"✅ Results saved to {output_json}")
        
        if output_video:
            print(f"✅ Video saved to {output_video}")
        
        print(f"✅ Processed {frame_count} frames")
        return all_detections
    
    def batch_process_images(self, folder_path, output_json=None):
        """Process all images in folder"""
        print(f"📁 Batch processing images from: {folder_path}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_detections = []
        
        for file in Path(folder_path).iterdir():
            if file.suffix.lower() in image_extensions:
                detections = self.process_image(str(file))
                if detections is not None:
                    all_detections.append({
                        'file': file.name,
                        'detections': detections
                    })
        
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(all_detections, f, indent=2)
            print(f"✅ Results saved to {output_json}")
        
        return all_detections
    
    def cap_read(self, cap):
        """Helper for camera read"""
        return cap.read()


def main():
    parser = argparse.ArgumentParser(description='CWAZZY VISION - Face Detection System')
    parser.add_argument('--mode', type=str, default='webcam',
                       choices=['webcam', 'image', 'video', 'batch'],
                       help='Mode: webcam, image, video, or batch')
    parser.add_argument('--source', type=str, help='Source file or folder path')
    parser.add_argument('--output-json', type=str, help='Output JSON file path')
    parser.add_argument('--output-video', type=str, help='Output video file path')
    
    args = parser.parse_args()
    
    # Initialize
    system = CWazzyVision()
    
    # Process based on mode
    if args.mode == 'webcam':
        system.process_webcam(output_json=args.output_json)
    
    elif args.mode == 'image':
        if not args.source:
            print("❌ Please provide --source for image mode")
            return
        system.process_image(args.source, output_json=args.output_json)
    
    elif args.mode == 'video':
        if not args.source:
            print("❌ Please provide --source for video mode")
            return
        system.process_video(args.source, output_json=args.output_json, 
                            output_video=args.output_video)
    
    elif args.mode == 'batch':
        if not args.source:
            print("❌ Please provide --source (folder) for batch mode")
            return
        system.batch_process_images(args.source, output_json=args.output_json)


if __name__ == "__main__":
    main()