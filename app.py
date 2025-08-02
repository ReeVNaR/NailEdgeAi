import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Changed from 1 to 2
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FINGERTIP_INDICES = [4, 8, 12, 16, 20]

# Load nail image with error handling
try:
    NAIL_IMG_PATH = os.path.join("Nails", "nail.png")
    if not os.path.exists(NAIL_IMG_PATH):
        logger.error(f"Nail image not found at {NAIL_IMG_PATH}")
        nail_png = None
    else:
        nail_png = cv2.imread(NAIL_IMG_PATH, cv2.IMREAD_UNCHANGED)
        if nail_png is None:
            logger.error("Failed to load nail image")
except Exception as e:
    logger.error(f"Error loading nail image: {e}")
    nail_png = None

def rotate_image(image, angle):
    """Rotate an RGBA image around its center."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    if image.shape[2] == 4:  # Preserve alpha channel
        alpha_channel = image[:, :, 3]
        rotated_alpha = cv2.warpAffine(alpha_channel, rot_matrix, (w, h))
        rotated = np.dstack((rotated[:, :, :3], rotated_alpha))
    return rotated

def overlay_image(background, overlay, x, y, overlay_size=40, angle=0):
    """Overlay PNG with rotation at given coordinates."""
    if overlay is None:
        return
    h, w = background.shape[:2]

    # Resize and rotate overlay
    overlay_resized = cv2.resize(overlay, (overlay_size, overlay_size))
    overlay_rotated = rotate_image(overlay_resized, angle)

    oh, ow = overlay_rotated.shape[:2]
    x1, y1 = max(0, x - ow // 2), max(0, y - oh // 2)
    x2, y2 = min(w, x1 + ow), min(h, y1 + oh)
    overlay_cropped = overlay_rotated[0:(y2 - y1), 0:(x2 - x1)]

    # Blend overlay with alpha
    if overlay_cropped.shape[2] == 4:
        alpha = overlay_cropped[:, :, 3] / 255.0
        for c in range(3):
            background[y1:y2, x1:x2, c] = (
                (1 - alpha) * background[y1:y2, x1:x2, c] +
                alpha * overlay_cropped[:, :, c]
            )
    else:
        background[y1:y2, x1:x2] = overlay_cropped

def get_hand_size(hand_landmarks, image_width, image_height):
    """Calculate relative hand size based on distance between wrist and middle finger tip"""
    wrist = hand_landmarks.landmark[0]
    middle_finger_tip = hand_landmarks.landmark[12]
    
    # Calculate distance in pixels
    wrist_to_tip = math.sqrt(
        (wrist.x - middle_finger_tip.x) ** 2 * image_width ** 2 +
        (wrist.y - middle_finger_tip.y) ** 2 * image_height ** 2
    )
    
    # Convert to relative size (40 is our base nail size)
    return int(wrist_to_tip * 0.18)  # Adjust multiplier as needed

def is_palm_facing_camera(hand_landmarks, handedness):
    """Determine if palm is facing the camera using hand landmarks and hand type"""
    # Get three non-collinear points to form a plane
    wrist = np.array([
        hand_landmarks.landmark[0].x,
        hand_landmarks.landmark[0].y,
        hand_landmarks.landmark[0].z
    ])
    index_mcp = np.array([
        hand_landmarks.landmark[5].x,
        hand_landmarks.landmark[5].y,
        hand_landmarks.landmark[5].z
    ])
    pinky_mcp = np.array([
        hand_landmarks.landmark[17].x,
        hand_landmarks.landmark[17].y,
        hand_landmarks.landmark[17].z
    ])

    # Calculate vectors for plane
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist

    # Calculate normal vector of palm plane
    normal = np.cross(v1, v2)
    
    # Adjust for hand type (left or right)
    is_right_hand = handedness.classification[0].label == "Right"
    
    # For right hand, positive Z means palm is visible
    # For left hand, negative Z means palm is visible
    return normal[2] > 0 if is_right_hand else normal[2] < 0

def process_frame(frame, filter_on=True, nail_size=40, rotation_offset=0):
    try:
        # Input validation
        if not isinstance(frame, str):
            logger.error("Invalid frame format")
            return None
            
        if ',' not in frame:
            logger.error("Invalid frame data format")
            return None

        # Decode image with error handling
        try:
            nparr = np.frombuffer(base64.b64decode(frame.split(',')[1]), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

        if image is None or image.size == 0:
            logger.error("Empty or invalid image")
            return None

        # Check image size to prevent memory issues
        if image.shape[0] * image.shape[1] > 4096 * 2160:  # 4K resolution limit
            logger.warning("Image too large, resizing")
            scale = min(4096/image.shape[1], 2160/image.shape[0])
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                h, w, _ = image.shape
                
                # Only show nails when back of hand is visible
                if not is_palm_facing_camera(hand_landmarks, handedness):
                    # Calculate adaptive nail size based on hand distance
                    adaptive_size = get_hand_size(hand_landmarks, w, h)
                    # Scale the user-selected size proportionally
                    final_nail_size = int(adaptive_size * (nail_size / 40))
                    
                    for i in FINGERTIP_INDICES:
                        # Find base joint index
                        base_idx = 3 if i == 4 else i - 2

                        tip_x = int(hand_landmarks.landmark[i].x * w)
                        tip_y = int(hand_landmarks.landmark[i].y * h)
                        base_x = int(hand_landmarks.landmark[base_idx].x * w)
                        base_y = int(hand_landmarks.landmark[base_idx].y * h)

                        dx = tip_x - base_x
                        dy = tip_y - base_y
                        angle = math.degrees(math.atan2(dy, dx))

                        if filter_on and nail_png is not None:
                            overlay_image(
                                image,
                                nail_png,
                                tip_x,
                                tip_y,
                                overlay_size=final_nail_size,
                                angle=-(angle) + rotation_offset
                            )
                        else:
                            radius = max(5, final_nail_size // 3)
                            cv2.circle(image, (tip_x, tip_y), radius, (255, 0, 255), -1)

        _, buffer = cv2.imencode('.jpg', image)
        if buffer is None:
            logger.error("Failed to encode processed image")
            return None
            
        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        logger.error(f"Error in processing frame: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400

        frame = data.get('frame')
        if not frame:
            return jsonify({'error': 'No frame data'}), 400

        filter_on = bool(data.get('filter', True))
        
        try:
            nail_size = int(data.get('size', 40))
            nail_size = max(20, min(nail_size, 100))  # Clamp between 20 and 100
        except (ValueError, TypeError):
            nail_size = 40

        try:
            rotation_offset = int(data.get('rotation', 0))
            rotation_offset = max(-180, min(rotation_offset, 180))  # Clamp between -180 and 180
        except (ValueError, TypeError):
            rotation_offset = 0

        processed_frame = process_frame(frame, filter_on, nail_size, rotation_offset)
        if processed_frame is None:
            return jsonify({'error': 'Frame processing failed'}), 400
            
        return jsonify({'frame': processed_frame})

    except Exception as e:
        logger.error(f"Route error: {e}")
        return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    # Generate self-signed certificate if not exists
    cert_path = 'cert.pem'
    key_path = 'key.pem'
    
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        from OpenSSL import crypto
        
        # Generate key
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        
        # Generate certificate
        cert = crypto.X509()
        cert.get_subject().CN = "localhost"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for one year
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha256')
        
        # Save certificate and private key
        with open(cert_path, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(key_path, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    # Run with SSL
    app.run(debug=True, host='0.0.0.0', ssl_context=(cert_path, key_path))
