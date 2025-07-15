# utils.py
import base64
import io
import cv2
import numpy as np
from PIL import Image

# --- Bounding Box Colors ---
COLOR_PERSON_WITH_ID = (0, 200, 0)
COLOR_RECOGNIZED_NO_ID = (255, 140, 0)
COLOR_UNKNOWN_NO_ID = (255, 215, 0)
COLOR_ID_CARD = (30, 144, 255)
COLOR_TEXT = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0) # Default background for text

def decode_image(base64_string):
    """Decodes a base64 string (potentially with data URI prefix) to an OpenCV image."""
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(',')[1]
        img_bytes = base64.b64decode(base64_string)
        img_pil = Image.open(io.BytesIO(img_bytes))
        # Convert to BGR for OpenCV, handling grayscale images
        if img_pil.mode == 'RGB':
            img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        elif img_pil.mode == 'RGBA': # Handle transparency if needed
             img_cv2_rgba = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
             img_cv2 = cv2.cvtColor(img_cv2_rgba, cv2.COLOR_BGRA2BGR) # Convert to BGR
        elif img_pil.mode == 'L': # Grayscale
            img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_GRAY2BGR)
        else: # Fallback for other modes
            img_cv2 = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        return img_cv2
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def encode_image(frame, quality=85):
    """Encodes an OpenCV frame (numpy array) to a base64 string (JPEG format)."""
    if frame is None:
        return None
    try:
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise ValueError("cv2.imencode failed")
        # Return only the base64 part, without the data URI prefix
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None


def draw_text_with_background(img, text, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                              color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), alpha=0.6, padding=3):
    """Draws text with a semi-transparent background rectangle."""
    try:
        (text_w, text_h), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
        x, y = org

        # Calculate rectangle coordinates carefully, ensuring they are within image bounds
        rect_x1 = max(0, x - padding)
        rect_y1 = max(0, y - text_h - baseline - padding)
        rect_x2 = min(img.shape[1], x + text_w + padding)
        rect_y2 = min(img.shape[0], y + baseline + padding)

        # Check if the rectangle dimensions are valid
        if rect_x1 >= rect_x2 or rect_y1 >= rect_y2:
            # If rect is invalid, just draw text without background (or skip)
            # cv2.putText(img, text, (x, y), fontFace, fontScale, color, thickness, lineType=cv2.LINE_AA)
            # print(f"Warning: Invalid background rect for text '{text}'. Skipping background.")
            pass # Skip drawing if rect is invalid

        else:
            sub_img = img[rect_y1:rect_y2, rect_x1:rect_x2]

            # Ensure sub_img is not empty (can happen with edge cases)
            if sub_img.size == 0:
                 # cv2.putText(img, text, (x, y), fontFace, fontScale, color, thickness, lineType=cv2.LINE_AA)
                 # print(f"Warning: Empty sub-image for text '{text}'. Skipping background.")
                 pass # Skip drawing if sub-image is empty
            else:
                # Create background rectangle
                bg_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)

                # Blend background with original image area
                res = cv2.addWeighted(sub_img, 1.0 - alpha, bg_rect, alpha, 1.0)

                # Put the blended region back onto the main image
                img[rect_y1:rect_y2, rect_x1:rect_x2] = res

        # Draw the text on top
        # Adjust text position slightly if background was drawn to align with original 'org'
        text_org_x = x
        text_org_y = y
        cv2.putText(img, text, (text_org_x, text_org_y), fontFace, fontScale, color, thickness, lineType=cv2.LINE_AA)

    except Exception as e:
        print(f"Error drawing text '{text}': {e}")
        # Fallback: Try drawing text without background if error occurs
        try:
            cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType=cv2.LINE_AA)
        except Exception as fallback_e:
            print(f"Error drawing text '{text}' (fallback failed): {fallback_e}")