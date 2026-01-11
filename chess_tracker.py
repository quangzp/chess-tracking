"""
Integrated chess board calibration and piece tracking system.
Combines board calibration with YOLO11-based piece detection and move tracking.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from piece_detector import ChessPieceDetector


class ChessBoardTracker:
    """Complete chess board tracking system with calibration and piece detection."""
    
    def __init__(self, sqdict_path='sqdict.json', model_path='yolov8n.pt', cam_rot=0, device='cpu'):
        """
        Initialize chess board tracker.
        
        Args:
            sqdict_path: Path to calibration JSON
            model_path: Path to YOLO model weights
            cam_rot: Camera rotation (0, 90, 180, 270)
            device: Device to use ('cpu' or 'cuda:0', etc.)
        """
        self.sqdict_path = sqdict_path
        self.cam_rot = cam_rot
        self.device = device
        
        # Check if calibration exists
        if not Path(sqdict_path).exists():
            raise FileNotFoundError(
                f"Calibration file not found: {sqdict_path}\n"
                f"Please run calibrate_manual_oriented.py first to generate it."
            )
        
        # Load calibration
        with open(sqdict_path, 'r') as f:
            self.sqdict = json.load(f)
        
        # Initialize piece detector
        print("[INFO] Loading YOLO11 model...")
        self.detector = ChessPieceDetector(sqdict_path=sqdict_path, model_name=model_path, device=device)
        
        self.move_history = []
        self.frame_count = 0
        
    def add_board_overlay(self, frame):
        """
        Draw calibrated board overlay on frame with full 9x9 grid.
        Draws all squares from sqdict directly.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with board overlay (grid + square labels)
        """
        vis = frame.copy()
        
        files = 'abcdefgh'
        ranks = '87654321'
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw each square from sqdict
        for square_name, square_coords in self.sqdict.items():
            polygon = np.array(square_coords, dtype=np.int32)
            
            # Draw square outline
            cv2.polylines(vis, [polygon], True, (180, 180, 180), 1)
            
            # Draw square label in center
            center = np.mean(square_coords, axis=0)
            cx, cy = int(center[0]), int(center[1])
            cv2.putText(vis, square_name, (cx - 12, cy + 5),
                       font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        
        return vis
    
    def process_frame(self, frame, show_board_overlay=True):
        """
        Process frame: add overlay, detect pieces, track moves.
        
        Args:
            frame: Input frame
            show_board_overlay: Whether to show calibrated board
            
        Returns:
            Dictionary with results and annotated frame
        """
        self.frame_count += 1
        
        # Add board overlay
        if show_board_overlay:
            frame = self.add_board_overlay(frame)
        
        # Detect pieces
        result = self.detector.process_frame(frame, draw=True)
        
        # Record moves
        for move in result['moves']:
            move_with_timestamp = {**move, 'frame': self.frame_count}
            self.move_history.append(move_with_timestamp)
            print(f"[FRAME {self.frame_count}] {move['from']} -> {move['to']} "
                  f"({move['piece']}) [{move['type']}]")
        
        return result
    
    def get_board_state(self):
        """
        Get current board state (which pieces are on which squares).
        
        Returns:
            Dictionary mapping square names to piece info
        """
        return self.detector.previous_positions.copy()
    
    def get_move_history(self):
        """Get all detected moves since initialization."""
        return self.move_history.copy()
    
    def save_session(self, output_file='session_data.json'):
        """
        Save session data (move history and final board state).
        
        Args:
            output_file: Path to output JSON file
        """
        session_data = {
            'total_frames': self.frame_count,
            'total_moves': len(self.move_history),
            'move_history': self.move_history,
            #'final_board_state': self.get_board_state(),
            'camera_rotation': self.cam_rot,
            'calibration_file': self.sqdict_path
        }
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\n[✅] Session saved to: {output_file}")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Total moves detected: {len(self.move_history)}")


def main():
    parser = argparse.ArgumentParser(
        description='Chess board tracking with YOLO11 piece detection'
    )
    parser.add_argument('--sqdict', type=str, default='sqdict.json',
                       help='Path to calibration JSON file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270],
                       help='Camera rotation (0=front, 90=right, 180=back, 270=left)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for piece detection')
    parser.add_argument('--save', type=str, default='session_data.json',
                       help='Output file to save session data')
    parser.add_argument('--no-overlay', action='store_true',
                       help='Disable board overlay')
    
    args = parser.parse_args()
    
    print("[INFO] Initializing Chess Board Tracker...")
    tracker = ChessBoardTracker(
        sqdict_path=args.sqdict,
        model_path=args.model,
        cam_rot=args.rotate
    )
    
    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return
    
    cv2.namedWindow("Chess Board Tracking")
    
    print("\n[📸] Controls:")
    print("  'q' - Quit and save session")
    print("  's' - Print current board state")
    print("  'c' - Clear move history")
    print("  'p' - Print move history")
    print("\n[INFO] Starting chess tracking...\n")
    
    # Set confidence threshold in detector
    tracker.detector.model.conf = args.confidence
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            result = tracker.process_frame(frame, show_board_overlay=not args.no_overlay)
            
            # Display frame
            vis = result['frame']
            
            # Add info overlay
            info_text = f"Frame: {tracker.frame_count} | Moves: {len(tracker.move_history)}"
            cv2.putText(vis, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Chess Board Tracking", vis)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[INFO] Quitting and saving session...")
                tracker.save_session(args.save)
                break
            elif key == ord('s'):
                board_state = tracker.get_board_state()
                print("\n[BOARD STATE]")
                if board_state:
                    for square in sorted(board_state.keys()):
                        piece = board_state[square]
                        print(f"  {square}: {piece['class']} (conf: {piece['confidence']:.2f})")
                else:
                    print("  No pieces detected")
                print()
            elif key == ord('c'):
                tracker.move_history = []
                print("[INFO] Move history cleared\n")
            elif key == ord('p'):
                history = tracker.get_move_history()
                print("\n[MOVE HISTORY]")
                if history:
                    for i, move in enumerate(history, 1):
                        print(f"  {i}. {move['from']} -> {move['to']} "
                              f"({move['piece']}) [Frame {move['frame']}]")
                else:
                    print("  No moves recorded")
                print()
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Saving session...")
        tracker.save_session(args.save)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
