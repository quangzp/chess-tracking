import cv2
import numpy as np
import json
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict


class ChessPieceDetector:
    """Detect and track chess pieces using YOLO11 with calibrated board coordinates."""
    
    def __init__(self, sqdict_path='sqdict.json', model_name='yolo_model/best.pt', device='cpu'):
        """
        Initialize the piece detector.
        
        Args:
            sqdict_path: Path to calibration JSON file containing square coordinates
            model_name: Path to YOLO model weights (default: yolo_model/best.pt)
            device: Device to use ('cpu' or 'cuda:0', etc.)
        """
        self.sqdict_path = sqdict_path
        self.device = device
        self.model = YOLO(model_name)
        self.model.to(device)  # Set device
        self.sqdict = self._load_sqdict()
        self.piece_classes = self._setup_piece_classes()
        self.previous_positions = {}  # Track previous piece positions
        
    def _load_sqdict(self):
        """Load the calibration data (square coordinates)."""
        if not Path(self.sqdict_path).exists():
            raise FileNotFoundError(f"Calibration file not found: {self.sqdict_path}")
        
        with open(self.sqdict_path, 'r') as f:
            return json.load(f)
    
    def _setup_piece_classes(self):
        """Setup chess piece class mapping. Requires training or using a chess-specific model."""
        # Standard COCO classes contain person, horse, dog, etc.
        # For production, you should train YOLO on chess piece dataset
        piece_mapping = {
            'pawn': 'pawn',
            'knight': 'knight',  
            'bishop': 'bishop',
            'rook': 'rook',
            'queen': 'queen',
            'king': 'king'
        }
        return piece_mapping
    
    def detect_pieces(self, frame, confidence_threshold=0.5, persist_tracks=True):
        """
        Track chess pieces in the frame using YOLO11 tracking.
        
        Args:
            frame: Input image/frame
            confidence_threshold: Minimum confidence for detections
            persist_tracks: Whether to persist tracks across frames
            
        Returns:
            List of detected/tracked pieces with track IDs and positions
        """
        # Use track() instead of predict() for continuous tracking
        results = self.model.track(frame, conf=confidence_threshold, persist=persist_tracks, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            # Check if tracking data is available
            if boxes.id is not None:  # Tracking mode
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                    class_name = result.names[class_id]
                    
                    # Calculate center of bounding box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'confidence': float(confidence),
                        'class': class_name,
                        'class_id': class_id,
                        'track_id': track_id  # Unique tracking ID
                    })
            else:  # Fallback to detection mode if tracking not available
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'confidence': float(confidence),
                        'class': class_name,
                        'class_id': class_id,
                        'track_id': -1  # No tracking ID in detection mode
                    })
        
        return detections
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon (square)."""
        x, y = point
        polygon = np.array(polygon)
        
        # Use cv2.pointPolygonTest for accuracy
        result = cv2.pointPolygonTest(polygon.astype(np.int32), (x, y), False)
        return result >= 0
    
    def map_pieces_to_squares(self, detections):
        """
        Map tracked pieces to board squares.
        
        Args:
            detections: List of detected/tracked pieces from detect_pieces()
            
        Returns:
            Dictionary mapping square names (e.g., 'a1') to piece information with track_id
        """
        piece_positions = {}
        
        for detection in detections:
            center = detection['center']
            
            # Find which square contains this piece
            for square_name, square_coords in self.sqdict.items():
                if self.point_in_polygon(center, square_coords):
                    piece_positions[square_name] = {
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'center': center,
                        'bbox': detection['bbox'],
                        'track_id': detection['track_id']  # Include track ID
                    }
                    break
        
        return piece_positions
    
    def detect_moves(self, current_positions):
        """
        Detect piece movements by comparing with previous positions using track IDs.
        
        Args:
            current_positions: Dictionary from map_pieces_to_squares()
            
        Returns:
            List of detected moves with track IDs
        """
        moves = []
        
        # Build mappings from track_id for more reliable tracking
        current_track_map = {}  # track_id -> (square, piece_info)
        for square, piece_info in current_positions.items():
            track_id = piece_info['track_id']
            if track_id != -1:  # Valid tracking ID
                current_track_map[track_id] = (square, piece_info)
        
        prev_track_map = {}
        for square, piece_info in self.previous_positions.items():
            track_id = piece_info.get('track_id', -1)
            if track_id != -1:
                prev_track_map[track_id] = (square, piece_info)
        
        # Detect moves using track IDs (more reliable)
        for track_id, (curr_square, curr_piece) in current_track_map.items():
            if track_id in prev_track_map:
                prev_square, prev_piece = prev_track_map[track_id]
                if curr_square != prev_square:
                    # Same piece moved to different square
                    moves.append({
                        'from': prev_square,
                        'to': curr_square,
                        'piece': curr_piece['class'],
                        'track_id': track_id,
                        'type': 'move'
                    })
        
        # Fallback: detect by piece type if track_id not available
        for square, piece_info in current_positions.items():
            if piece_info['track_id'] == -1:  # No valid track ID
                if square not in self.previous_positions:
                    # Check if piece disappeared from another square
                    for prev_square, prev_piece in self.previous_positions.items():
                        if prev_piece['class'] == piece_info['class'] and prev_square not in current_positions:
                            moves.append({
                                'from': prev_square,
                                'to': square,
                                'piece': piece_info['class'],
                                'track_id': -1,
                                'type': 'move'
                            })
                            break
        
        self.previous_positions = current_positions.copy()
        return moves
    
    def draw_detections(self, frame, detections, piece_positions):
        """
        Draw detection boxes with track IDs and square labels on frame.
        
        Args:
            frame: Input frame
            detections: List from detect_pieces()
            piece_positions: Dictionary from map_pieces_to_squares()
            
        Returns:
            Annotated frame
        """
        vis = frame.copy()
        
        # Draw bounding boxes for detections with track IDs
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection['track_id']
            
            # Different color for tracked vs untracked
            color = (0, 255, 0) if track_id != -1 else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Label with track ID
            if track_id != -1:
                label = f"ID:{track_id} {detection['class']} {detection['confidence']:.2f}"
            else:
                label = f"{detection['class']} {detection['confidence']:.2f}"
            
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw square labels with detected pieces
        for square_name, piece_info in piece_positions.items():
            square_coords = self.sqdict[square_name]
            polygon = np.array(square_coords, dtype=np.int32)
            
            # Highlight squares with pieces
            cv2.polylines(vis, [polygon], True, (255, 0, 0), 2)
            
            # Draw piece label on square with track ID
            center = piece_info['center']
            track_id = piece_info['track_id']
            if track_id != -1:
                label = f"{square_name}\n{piece_info['class']}\nID:{track_id}"
            else:
                label = f"{square_name}\n{piece_info['class']}"
            
            cv2.putText(vis, label, (int(center[0]) - 20, int(center[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return vis
    
    def process_frame(self, frame, draw=True):
        """
        Complete pipeline: detect pieces, map to squares, and track moves.
        
        Args:
            frame: Input frame
            draw: Whether to draw annotations on frame
            
        Returns:
            Dictionary containing:
                - 'positions': piece positions on board
                - 'moves': detected moves
                - 'frame': annotated frame (if draw=True)
        """
        detections = self.detect_pieces(frame)
        positions = self.map_pieces_to_squares(detections)
        moves = self.detect_moves(positions)
        
        result = {
            'positions': positions,
            'moves': moves,
            'detections': detections
        }
        
        if draw:
            result['frame'] = self.draw_detections(frame, detections, positions)
        
        return result


def main():
    """Example usage of ChessPieceDetector."""
    print("[INFO] Initializing Chess Piece Detector...")
    
    detector = ChessPieceDetector(sqdict_path='sqdict.json')
    
    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return
    
    cv2.namedWindow("Chess Piece Tracking")
    
    print("[INFO] Press 'q' to exit, 's' to print current positions")
    print("[INFO] Starting piece detection...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process frame with piece detection
        result = detector.process_frame(frame, draw=True)
        
        positions = result['positions']
        moves = result['moves']
        
        # Display moves if any detected
        if moves:
            print(f"[MOVE DETECTED]")
            for move in moves:
                print(f"  {move['from']} -> {move['to']} ({move['piece']}) [{move['type']}]")
        
        # Show frame with detections
        vis = result['frame']
        cv2.imshow("Chess Piece Tracking", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Exiting...")
            break
        elif key == ord('s'):
            print("\n[CURRENT POSITIONS]")
            for square, piece_info in sorted(positions.items()):
                print(f"  {square}: {piece_info['class']} (confidence: {piece_info['confidence']:.2f})")
            print()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
