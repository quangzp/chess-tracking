import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from chess_tracker import ChessBoardTracker


def run_calibration(cap, cam_rot=0):
    """
    Run the board calibration process using existing camera object.
    
    Args:
        cap: Existing cv2.VideoCapture object
        cam_rot: Camera rotation (0, 90, 180, 270)
    
    Returns:
        True if calibration successful and saved, False otherwise
    """
    print("\n" + "="*60)
    print("[📸 STEP 1: BOARD CALIBRATION]")
    print("="*60)
    print(f"Camera rotation: {cam_rot}° (CW)")
    print("\nInstructions:")
    print(" 1️⃣ Point the camera to the board position")
    print(" 2️⃣ Click 4 corner points (order: top-left, top-right, bottom-right, bottom-left)")
    print(" 3️⃣ Press 'r' to reset, 's' to save, 'q' to quit without saving")
    print("[INFO] Using existing camera connection (no restart)\n")
    
    points = []
    
    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                print(f"[INFO] Point {len(points)}: ({x}, {y})")
            else:
                print("[INFO] 4 points collected. Press 'r' to reset or 's' to save.")
    
    def remap_index(r_disp, c_disp, cam_rot):
        """Remap index from camera view to standard board notation."""
        if cam_rot == 0:
            r_std, c_std = r_disp, c_disp
        elif cam_rot == 90:
            r_std, c_std = c_disp, 7 - r_disp
        elif cam_rot == 180:
            r_std, c_std = 7 - r_disp, 7 - c_disp
        elif cam_rot == 270:
            r_std, c_std = 7 - c_disp, r_disp
        else:
            r_std, c_std = r_disp, c_disp
        return r_std, c_std
    
    cv2.namedWindow("Board Calibration")
    cv2.setMouseCallback("Board Calibration", mouse_click)
    
    calibration_done = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        vis = frame.copy()
        
        # Draw clicked points
        for idx, p in enumerate(points):
            cv2.circle(vis, p, 6, (0, 0, 255), -1)
            cv2.putText(vis, str(idx+1), (p[0]+8, p[1]-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(points) == 4:
            # Draw box & grid
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (255, 255, 255), 2)
            
            src = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
            dst = np.array(points, dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            
            src_grid = np.array([[[x, y] for x in range(9)] for y in range(9)], dtype=np.float32)
            dst_grid = cv2.perspectiveTransform(src_grid.reshape(-1, 1, 2), H).reshape(9, 9, 2)
            
            # Draw grid
            for r in range(9):
                cv2.polylines(vis, [dst_grid[r, :, :].astype(int)], False, (180, 180, 180), 1)
            for c in range(9):
                cv2.polylines(vis, [dst_grid[:, c, :].astype(int)], False, (180, 180, 180), 1)
            
            # Draw square labels
            files = 'abcdefgh'
            ranks = '87654321'
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            for r in range(8):
                for c in range(8):
                    r_std, c_std = remap_index(r, c, cam_rot)
                    file_letter = files[c_std]
                    rank_char = ranks[r_std]
                    label = f"{file_letter}{rank_char}"
                    
                    center = dst_grid[r, c] + (dst_grid[r+1, c+1] - dst_grid[r, c]) / 2
                    cx, cy = int(center[0]), int(center[1])
                    cv2.putText(vis, label, (cx-12, cy+5), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Add status text
        status_text = f"Points: {len(points)}/4"
        cv2.putText(vis, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Board Calibration", vis)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n[INFO] Calibration cancelled (no save).")
            break
        elif key == ord('r'):
            points = []
            print("[INFO] Points reset.")
        elif key == ord('s'):
            if len(points) != 4:
                print("[WARN] Must click 4 points before saving.")
                continue
            
            # Process and save calibration
            src = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
            dst = np.array(points, dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            
            src_grid = np.array([[[x, y] for x in range(9)] for y in range(9)], dtype=np.float32)
            dst_grid = cv2.perspectiveTransform(src_grid.reshape(-1, 1, 2), H).reshape(9, 9, 2)
            
            displayed_squares = {}
            for r in range(8):
                for c in range(8):
                    tl = dst_grid[r, c].tolist()
                    tr = dst_grid[r, c+1].tolist()
                    br = dst_grid[r+1, c+1].tolist()
                    bl = dst_grid[r+1, c].tolist()
                    displayed_squares[(r, c)] = [tl, tr, br, bl]
            
            # Remap to standard notation
            files = 'abcdefgh'
            ranks = '87654321'
            squares_std = {}
            for (r_disp, c_disp), poly in displayed_squares.items():
                r_std, c_std = remap_index(r_disp, c_disp, cam_rot)
                file_letter = files[c_std]
                rank_char = ranks[r_std]
                squares_std[f"{file_letter}{rank_char}"] = poly
            
            with open('sqdict.json', 'w') as f:
                json.dump(squares_std, f, indent=2)
            
            print(f"\n[✅] Calibration saved to sqdict.json")
            print(f"    Board rotation: {cam_rot}° (CW)")
            calibration_done = True
            break
    
    cv2.destroyAllWindows()
    
    return calibration_done


def run_tracking(cap, sqdict_path='sqdict.json', model_path='yolo_model/best.pt', cam_rot=0, confidence=0.5, device='cpu', frame_skip=1):
    """
    Run the piece tracking system using existing camera object.
    
    Args:
        cap: Existing cv2.VideoCapture object
        sqdict_path: Path to calibration JSON
        model_path: Path to YOLO model
        cam_rot: Camera rotation
        confidence: Detection confidence threshold
        device: Device to use ('cpu' or 'cuda')
        frame_skip: Process every N frames
    """
    print("\n" + "="*60)
    print("[🤖 STEP 2: PIECE TRACKING WITH YOLO11]")
    print("="*60)
    print(f"Calibration file: {sqdict_path}")
    print(f"YOLO model: {model_path}")
    print(f"Camera rotation: {cam_rot}°")
    print(f"Confidence threshold: {confidence}")
    print(f"Device: {device}")
    print(f"Frame skip: {frame_skip}\n")
    
    try:
        print("[INFO] Initializing tracker...")
        tracker = ChessBoardTracker(
            sqdict_path=sqdict_path,
            device=device,
            model_path=model_path,
            cam_rot=cam_rot
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print("[INFO] Using existing camera connection (no restart)\n")
    
    # Camera already opened and configured
    
    cv2.namedWindow("Chess Piece Tracking - Real-time")
    
    print("\n[🎮 Controls]")
    print("  'q' - Quit and save session")
    print("  's' - Print current board state")
    print("  'c' - Clear move history")
    print("  'p' - Print move history")
    print("  'g' - Toggle grid overlay")
    print("\n[INFO] Starting real-time tracking...\n")
    
    show_grid = True
    tracker.detector.model.conf = confidence
    frame_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_counter += 1
            
            # Skip frames for CPU optimization
            if frame_counter % frame_skip != 0:
                continue
            
            # Add grid overlay
            if show_grid:
                frame = tracker.add_board_overlay(frame)
            
            # Detect pieces
            result = tracker.detector.process_frame(frame, draw=True)
            vis = result['frame']
            
            positions = result['positions']
            moves = result['moves']
            
            tracker.frame_count += 1
            
            # Print moves
            if moves:
                for move in moves:
                    print(f"[FRAME {tracker.frame_count}] {move['from']} -> {move['to']} "
                          f"({move['piece']}) [{move['type'].upper()}]")
                    move_with_frame = {**move, 'frame': tracker.frame_count}
                    tracker.move_history.append(move_with_frame)
            
            # Draw info panel
            info_lines = [
                f"Frame: {tracker.frame_count}",
                f"Pieces: {len(positions)}",
                f"Moves: {len(tracker.move_history)}",
                f"Grid: {'ON' if show_grid else 'OFF'}",
                f"Device: {device}",
                f"Skip: 1/{frame_skip}"
            ]
            
            y_offset = 30
            for info in info_lines:
                cv2.putText(vis, info, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # Draw detected pieces
            if positions:
                y_offset += 10
                cv2.putText(vis, "Pieces detected:", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
                for square in sorted(positions.keys()):
                    piece = positions[square]
                    info = f"{square}: {piece['class']}"
                    cv2.putText(vis, info, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                    y_offset += 16
            
            cv2.imshow("Chess Piece Tracking - Real-time", vis)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[INFO] Quitting and saving session...")
                tracker.save_session('session_data.json')
                break
            
            elif key == ord('s'):
                board_state = tracker.get_board_state()
                print("\n" + "-"*50)
                print("[BOARD STATE]")
                if board_state:
                    for square in sorted(board_state.keys()):
                        piece = board_state[square]
                        print(f"  {square}: {piece['class']} ({piece['confidence']:.2f})")
                else:
                    print("  No pieces detected")
                print("-"*50)
            
            elif key == ord('c'):
                tracker.move_history = []
                print("[INFO] Move history cleared\n")
            
            elif key == ord('p'):
                history = tracker.get_move_history()
                print("\n" + "-"*50)
                print("[MOVE HISTORY]")
                if history:
                    for i, move in enumerate(history, 1):
                        print(f"  {i}. {move['from']} -> {move['to']} "
                              f"({move['piece']}) [Frame {move['frame']}]")
                else:
                    print("  No moves recorded")
                print("-"*50)
            
            elif key == ord('g'):
                show_grid = not show_grid
                status = "ON" if show_grid else "OFF"
                print(f"[INFO] Grid overlay: {status}\n")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Saving session...")
        tracker.save_session('session_data.json')
    
    finally:
        # Camera will be closed by main() after tracking completes
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Integrated Chess Tracking System - Calibration + YOLO11 Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270],
                       help='Camera rotation (0=front, 90=right, 180=back, 270=left)')
    parser.add_argument('--model', type=str, default='yolo_model/best.pt',
                       help='Path to YOLO model weights (default: yolo_model/best.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--skip-calib', action='store_true',
                       help='Skip calibration, go directly to tracking')
    parser.add_argument('--recalib', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every N frames (higher = faster but less accurate, useful for CPU)ict.json')
    parser.add_argument('--sqdict', type=str, default='sqdict.json')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  CHESS TRACKING SYSTEM - INTEGRATED WORKFLOW")
    print("="*60)
    
    sqdict_exists = Path(args.sqdict).exists()
    should_calibrate = False
    
    # Determine if calibration is needed
    if not sqdict_exists:
        print("\n[INFO] Calibration file not found: sqdict.json")
        should_calibrate = True
    elif args.recalib:
        print("\n[INFO] Forcing re-calibration...")
        should_calibrate = True
    elif not args.skip_calib:
        print(f"\n[INFO] Calibration file found: {args.sqdict}")
        print("[OPTION] Press 'c' to re-calibrate, or any other key to skip:")
        
        import select
        if sys.stdin in select.select([sys.stdin], [], [], 5)[0]:
            user_input = sys.stdin.read(1).lower()
            if user_input == 'c':
                should_calibrate = True
        else:
            print("(Skipping re-calibration)")
    
    # === Open camera once for entire workflow ===
    print("\n[INFO] Opening camera (will be used for entire session)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        sys.exit(1)
    
    # Set camera properties (1280x720, 30fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[✅] Camera ready!\n")
    
    try:
        # Run calibration if needed
        if should_calibrate:
            calib_success = run_calibration(cap, cam_rot=args.rotate)
            if not calib_success:
                print("\n[INFO] Calibration was not completed. Exiting.")
                sys.exit(0)
        
        # Run tracking
        if Path(args.sqdict).exists():
            run_tracking(
                cap=cap,
                sqdict_path=args.sqdict,
                model_path=args.model,
                cam_rot=args.rotate,
                confidence=args.confidence,
                device=args.device,
                frame_skip=args.frame_skip
            )
        else:
            print(f"\n❌ Error: Calibration file not found: {args.sqdict}")
            sys.exit(1)
    
    finally:
        # Close camera once after everything is done
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("  SYSTEM SHUTDOWN - GOODBYE!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
