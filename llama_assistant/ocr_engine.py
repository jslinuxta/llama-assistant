from paddleocr import PaddleOCR
from PIL import Image
import copy
import numpy as np
import time


def group_boxes_to_lines(bboxes, vertical_tolerance=5):
    """
    Groups bounding boxes into lines based on vertical alignment.

    Args:
        bboxes: List of bounding boxes [(xmin, ymin, xmax, ymax)].
        vertical_tolerance: Tolerance for vertical proximity to consider boxes in the same line.

    Returns:
        List of lines, where each line is a list of bounding boxes.
    """
    # Sort bounding boxes by ymin (top edge)
    bboxes = sorted(bboxes, key=lambda bbox: bbox[1])

    lines = []
    current_line = []
    current_ymin = None

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        # Check if starting a new line or current box is not vertically aligned
        if current_ymin is None or ymin > current_ymin + vertical_tolerance:
            # Save the current line if not empty
            if current_line:
                # Sort current line by xmin
                current_line.sort(key=lambda box: box[0])
                lines.append(current_line)
            # Start a new line
            current_line = [bbox]
            current_ymin = ymin
        else:
            # Add box to the current line
            current_line.append(bbox)

    # Add the last line if any
    if current_line:
        current_line.sort(key=lambda box: box[0])
        lines.append(current_line)

    return lines


def quad_to_rect(quad_boxes):
    """
    Converts a quadrilateral bounding box to a rectangular bounding box.

    Args:
        quad_boxes: List of 4 points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] representing the quadrilateral.

    Returns:
        List of rectangular bounding box (xmin, ymin, xmax, ymax).
    """
    result = []
    for quad_box in quad_boxes:
        quad_box = quad_box.astype(np.int32)
        # Extract all x and y coordinates
        x_coords = quad_box[:, 0]
        y_coords = quad_box[:, 1]

        # Find the enclosing rectangle
        xmin = np.min(x_coords)
        ymin = np.min(y_coords)
        xmax = np.max(x_coords)
        ymax = np.max(y_coords)

        result.append((xmin, ymin, xmax, ymax))

    return result


class OCREngine:
    def __init__(self):
        self.ocr = None

    def load_ocr(self, processign_thread=None):
        if self.ocr is None:
            if processign_thread:
                processign_thread.set_preloading(True, "Initializing OCR ....")

            self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
            time.sleep(1.2)

            if processign_thread:
                processign_thread.set_preloading(False, "...")

    def perform_ocr(self, img_path, streaming=False, processing_thread=None):
        self.load_ocr(processing_thread)
        img = np.array(Image.open(img_path).convert("RGB"))
        ori_im = img.copy()

        # text detection
        dt_boxes, _ = self.ocr.text_detector(img)
        if dt_boxes is None:
            return None

        img_crop_list = []
        dt_boxes = quad_to_rect(dt_boxes)

        # group boxes into lines
        lines = group_boxes_to_lines(dt_boxes)

        # return a generator if streaming
        if streaming:

            def generate_result():
                for boxes in lines:
                    img_crop_list = []
                    for bno in range(len(boxes)):
                        xmin, ymin, xmax, ymax = copy.deepcopy(boxes[bno])
                        img_crop = ori_im[ymin:ymax, xmin:xmax]
                        if any([dim <= 0 for dim in img_crop.shape[:2]]):
                            continue
                        img_crop_list.append(img_crop)
                    rec_res, _ = self.ocr.text_recognizer(img_crop_list)

                    line_text = ""
                    for rec_result in rec_res:
                        text, score = rec_result
                        if score >= self.ocr.drop_score:
                            line_text += text + " "

                        yield line_text + "\n"

            return generate_result()

        # non-streaming
        full_result = ""
        for boxes in lines:
            img_crop_list = []
            for bno in range(len(boxes)):
                xmin, ymin, xmax, ymax = copy.deepcopy(boxes[bno])
                img_crop = ori_im[ymin:ymax, xmin:xmax]
                if any([dim <= 0 for dim in img_crop.shape[:2]]):
                    continue
                img_crop_list.append(img_crop)
            rec_res, _ = self.ocr.text_recognizer(img_crop_list)

            line_text = ""
            for rec_result in rec_res:
                text, score = rec_result
                if score >= self.ocr.drop_score:
                    line_text += text + " "

            full_result += line_text + "\n"

        return full_result


ocr_engine = OCREngine()
