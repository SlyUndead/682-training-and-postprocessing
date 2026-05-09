import cv2
import numpy as np
from PIL import Image
import math
from datetime import datetime, timezone
from shapely.geometry import Polygon
import concurrent.futures
import itertools

def get_random_color():
    """Generate a random RGB color tuple with values between 0 and 1."""
    import random
    return (random.random(), random.random(), random.random())

def preprocess_image(image):
    """Preprocesses an image using normalization, denoising, and edge enhancement."""
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_image = clahe.apply(image)
    
    # Normalize
    img_float = contrast_enhanced_image.astype(np.float32) / 255.0
    normalized_image = cv2.normalize(img_float, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Edge enhancement
    sobel_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    # Convert normalized image back to 8-bit
    normalized_8bit = (normalized_image * 255).astype(np.uint8)
    
    # Blend
    processed_image = cv2.addWeighted(normalized_8bit, 0.8, gradient_magnitude, 0.2, 0)
    
    # Convert back to PIL Image
    return Image.fromarray(processed_image)

def process_single_model(args):
    """
    Process the image using a single model.
    Includes confidence scores for each annotation.
    """
    img, model, requires_preprocessing = args
    
    # Preprocess image if required
    if requires_preprocessing:
        img = preprocess_image(img)
    
    results = model(img)
    bounding_boxes = []
    segmentations = []
    
    # Process bounding boxes with confidence scores
    for i, (detection, class_id) in enumerate(zip(results[0].boxes, results[0].boxes.cls)):
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        class_name = results[0].names[int(class_id.item())]
        confidence = float(detection.conf[0].item()) if hasattr(detection, 'conf') else 0.0
        
        annotation = {
            "label": class_name,
            "bounding_box": [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x2, "y": y2},
                {"x": x1, "y": y2},
                {"x": x1, "y": y1}
            ],
            "confidence": confidence,
            "box_index": i
        }
        bounding_boxes.append(annotation)

    # Process segmentations with confidence scores
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        for i in range(len(results[0].masks)):
            # Get class name and skip if it's background
            cls_index = int(results[0].boxes.cls[i])
            cls = results[0].names[cls_index]
            if cls.lower() == 'background':
                continue
            
            # Get confidence score for this segmentation
            confidence = float(results[0].boxes.conf[i].item()) if hasattr(results[0].boxes, 'conf') else 0.0
            
            annotation = {
                "label": cls,
                "segmentation": [{"x": float(point[0]), "y": float(point[1])} for point in results[0].masks.xy[i]],
                "confidence": confidence,
                "box_index": i
            }
            segmentations.append(annotation)

    # Create final annotations, matching bounding boxes with segmentations when possible
    final_annotations = []
    segmentation_map = {seg["box_index"]: seg for seg in segmentations}
    
    for box in bounding_boxes:
        box_index = box["box_index"]
        
        # Get matching segmentation if available
        matching_segmentation = segmentation_map.get(box_index)
        
        annotation = {
            "label": box["label"],
            "bounding_box": box["bounding_box"],
            "segmentation": matching_segmentation["segmentation"] if matching_segmentation else [],
            "created_by": "Model v1.0.0",
            "created_on": datetime.now(timezone.utc).isoformat()
        }
        
        # Use segmentation confidence if available, otherwise use bounding box confidence
        if matching_segmentation:
            annotation["confidence"] = matching_segmentation["confidence"]
        else:
            annotation["confidence"] = box["confidence"]
        
        final_annotations.append(annotation)
        
        # Clean up temporary fields
        if "box_index" in annotation:
            del annotation["box_index"]
    
    return final_annotations

def process_parallel_models(img, model_group):
    """Process the image using multiple models in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Create list of (img, model, requires_preprocessing) tuples
        model_inputs = [(img, model, requires_prep) for model, requires_prep in model_group]
        # Execute all models in parallel
        results = list(executor.map(process_single_model, model_inputs))
    
    # Combine results from all models
    combined_annotations = list(itertools.chain.from_iterable(results))
    
    # Remove duplicates based on bounding box overlap and label
    filtered_annotations = remove_duplicates(combined_annotations)
    
    return filtered_annotations

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    b1 = [box1[0]["x"], box1[0]["y"], box1[2]["x"], box1[2]["y"]]
    b2 = [box2[0]["x"], box2[0]["y"], box2[2]["x"], box2[2]["y"]]
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def calculate_overlap_percentage(seg1, seg2):
    """
    Calculate the percentage of overlap between two segmentations.
    Returns the overlap as a percentage of the smaller polygon's area.
    """
    # Convert segmentation points to Shapely polygons
    poly1_points = [(point["x"], point["y"]) for point in seg1]
    poly2_points = [(point["x"], point["y"]) for point in seg2]
    
    # Create Shapely polygons
    polygon1 = Polygon(poly1_points)
    polygon2 = Polygon(poly2_points)
    
    # Handle invalid polygons
    if not polygon1.is_valid or not polygon2.is_valid:
        polygon1 = polygon1.buffer(0)
        polygon2 = polygon2.buffer(0)
        
        if not polygon1.is_valid or not polygon2.is_valid:
            return 0.0
    
    # Calculate individual areas
    area1 = polygon1.area
    area2 = polygon2.area
    
    # Calculate intersection area
    intersection_area = polygon1.intersection(polygon2).area
    
    # Use the smaller polygon's area as the denominator
    smaller_area = min(area1, area2)
    
    if smaller_area <= 0:
        return 0.0
    
    # Return as a percentage
    return (intersection_area / smaller_area) * 100.0

def remove_duplicates(annotations, iou_threshold=50, edge_threshold_percent=0.1):
    """
    Remove duplicate detections based on IoU and confidence.
    Also removes tooth number annotations that are too close to the image edges.
    """
    filtered = []
    used = set()
    
    # Find image dimensions based on annotations
    all_x_coords = []
    for ann in annotations:
        if "bounding_box" in ann:
            for point in ann["bounding_box"]:
                all_x_coords.append(point["x"])
        elif "segmentation" in ann:
            for point in ann["segmentation"]:
                all_x_coords.append(point["x"])
    
    if not all_x_coords:
        return annotations
    
    # Calculate image width
    min_x, max_x = min(all_x_coords), max(all_x_coords)
    image_width = max_x - min_x
    
    # Calculate edge thresholds
    left_edge_threshold = min_x + (image_width * edge_threshold_percent)
    right_edge_threshold = max_x - (image_width * edge_threshold_percent)
    
    # Check if we need to handle teeth numbers (1-32)
    def is_tooth_number(label):
        try:
            num = int(label)
            return 1 <= num <= 32
        except (ValueError, TypeError):
            return False
    
    # Check if we have any teeth numbers (1-32) in our annotations
    has_teeth_numbers = any(is_tooth_number(ann["label"]) for ann in annotations)
    
    # First, filter out tooth numbers that are too close to edges
    edge_filtered = []
    for i, ann in enumerate(annotations):
        if is_tooth_number(ann["label"]):
            # Get the leftmost and rightmost points of the annotation
            if "segmentation" in ann:
                x_coords = [point["x"] for point in ann["segmentation"]]
            elif "bounding_box" in ann:
                x_coords = [point["x"] for point in ann["bounding_box"]]
            else:
                edge_filtered.append(ann)
                continue
            
            leftmost = min(x_coords)
            rightmost = max(x_coords)
            
            # Skip if too close to edges
            if leftmost <= left_edge_threshold or rightmost >= right_edge_threshold:
                continue
        
        edge_filtered.append(ann)
    
    # Now process for duplicates
    for i, ann1 in enumerate(edge_filtered):
        if i in used:
            continue
            
        current_group = [ann1]
        
        for j, ann2 in enumerate(edge_filtered[i+1:], i+1):
            if j in used:
                continue
            
            # Skip if needed
            if not (ann1["label"] == ann2["label"] or 
                   (has_teeth_numbers and 
                    is_tooth_number(ann1["label"]) and 
                    is_tooth_number(ann2["label"]))):
                continue
                
            # Check if they overlap significantly
            if calculate_overlap_percentage(ann1["segmentation"], ann2["segmentation"]) > iou_threshold:
                current_group.append(ann2)
                used.add(j)
                
        # Select the annotation with the most detailed segmentation
        best_annotation = max(
            current_group,
            key=lambda x: (
                x.get("confidence", 0.0),
                len(x.get("segmentation", []))
            )
        )
        filtered.append(best_annotation)
        used.add(i)
        
    return filtered

def postprocess_teeth_numbers_bitewing(annotations, img_array):
    """
    Advanced postprocessing for teeth annotations in bitewing images.
    """
    # Identify teeth number annotations
    teeth_annotations = []
    for i, ann in enumerate(annotations):
        try:
            num = int(ann["label"])
            if 11 <= num <= 48:  # Valid range for quadrant notation
                teeth_annotations.append((i, ann, num))
        except (ValueError, TypeError):
            continue
    
    if not teeth_annotations:
        return annotations
    
    # Calculate image dimensions
    mid_y = img_array.shape[0] / 2
    mid_x = img_array.shape[1] / 2
    
    # Group teeth by upper and lower jaw
    upper_teeth = []
    lower_teeth = []
    
    for i, ann, num in teeth_annotations:
        center_y = sum(point["y"] for point in ann["segmentation"]) / len(ann["segmentation"])
        center_x = sum(point["x"] for point in ann["segmentation"]) / len(ann["segmentation"])
        
        quadrant = num // 10
        is_upper_quadrant = quadrant in [1, 2]
        is_upper_position = center_y < mid_y
        
        if is_upper_position:
            upper_teeth.append((i, ann, num, center_x, is_upper_quadrant))
        else:
            lower_teeth.append((i, ann, num, center_x, is_upper_quadrant))
    
    # Sort teeth by x-coordinate within each jaw
    upper_teeth.sort(key=lambda t: t[3])
    lower_teeth.sort(key=lambda t: t[3])
    
    # Convert FDI number to Universal number
    def fdi_to_universal(fdi_number):
        quadrant = fdi_number // 10
        position = fdi_number % 10
        
        if position < 1 or position > 8:
            return 0
            
        if quadrant == 1:  # Upper right
            return 9 - position
        elif quadrant == 2:  # Upper left
            return 8 + position
        elif quadrant == 3:  # Lower left
            return 25 - position
        elif quadrant == 4:  # Lower right
            return 24 + position
        return 0
    
    # Process teeth and apply corrections
    for teeth_group in [upper_teeth, lower_teeth]:
        if len(teeth_group) >= 2:
            positions = [num % 10 for _, _, num, _, _ in teeth_group]
            
            increasing = sum(1 for i in range(len(positions) - 1) if positions[i+1] > positions[i])
            decreasing = sum(1 for i in range(len(positions) - 1) if positions[i+1] < positions[i])
            
            if teeth_group == upper_teeth:
                correct_quadrant = 2 if increasing > decreasing else 1
            else:
                correct_quadrant = 3 if increasing > decreasing else 4
            
            for i, (idx, ann, num, _, _) in enumerate(teeth_group):
                position = num % 10
                corrected_fdi = correct_quadrant * 10 + position
                universal_number = fdi_to_universal(corrected_fdi)
                annotations[idx]["original_label"] = str(num)
                annotations[idx]["original_label_after_correction"] = str(corrected_fdi)
                annotations[idx]["model_label"] = annotations[idx]["label"]
                annotations[idx]["label"] = str(universal_number)
    
    return annotations

def postprocess_teeth_numbers_pariapical(annotations, img_array, pariapical_model_orientation):
    """
    Advanced postprocessing for teeth annotations in periapical images.
    """
    # Identify teeth number annotations
    teeth_annotations = []
    for i, ann in enumerate(annotations):
        try:
            num = int(ann["label"])
            if 11 <= num <= 48:
                teeth_annotations.append((i, ann, num))
        except (ValueError, TypeError):
            continue
    
    if not teeth_annotations:
        return annotations
    
    # Calculate x coordinates for sorting
    for i, ann, num in teeth_annotations:
        if "segmentation" in ann and ann["segmentation"]:
            ann["center_x"] = sum(point["x"] for point in ann["segmentation"]) / len(ann["segmentation"])
        elif "bounding_box" in ann:
            ann["center_x"] = sum(point["x"] for point in ann["bounding_box"]) / len(ann["bounding_box"])
    
    # Determine jaw type using periapical orientation model
    result = pariapical_model_orientation(img_array)
    top_class_index = result[0].probs.top1
    jaw_type = result[0].names[top_class_index]
    
    # Group all teeth
    all_teeth = [(i, ann, num, ann.get("center_x", 0)) for i, ann, num in teeth_annotations]
    all_teeth.sort(key=lambda t: t[3])
    
    # Convert FDI number to Universal number
    def fdi_to_universal(fdi_number):
        quadrant = fdi_number // 10
        position = fdi_number % 10
        
        if position < 1 or position > 8:
            return 0
            
        if quadrant == 1:  # Upper right
            return 9 - position
        elif quadrant == 2:  # Upper left
            return 8 + position
        elif quadrant == 3:  # Lower left
            return 25 - position
        elif quadrant == 4:  # Lower right
            return 24 + position
        return 0
    
    # Determine correct quadrant based on jaw type and tooth positions
    if len(all_teeth) >= 2:
        positions = [num % 10 for _, _, num, _ in all_teeth]
        
        increasing = sum(1 for i in range(len(positions) - 1) if positions[i+1] > positions[i])
        decreasing = sum(1 for i in range(len(positions) - 1) if positions[i+1] < positions[i])
        
        if jaw_type.lower() == "upper":
            correct_quadrant = 2 if increasing > decreasing else 1
        else:
            correct_quadrant = 3 if increasing > decreasing else 4
    
    elif len(all_teeth) == 1:
        idx, ann, num, center_x = all_teeth[0]
        
        # Find image dimensions
        all_x_coords = []
        for a in annotations:
            if "segmentation" in a and a["segmentation"]:
                for point in a["segmentation"]:
                    all_x_coords.append(point["x"])
            elif "bounding_box" in a:
                for point in a["bounding_box"]:
                    all_x_coords.append(point["x"])
        
        if all_x_coords:
            min_x, max_x = min(all_x_coords), max(all_x_coords)
            image_midpoint = (min_x + max_x) / 2
            is_right_side = center_x < image_midpoint
            
            if jaw_type.lower() == "upper":
                correct_quadrant = 1 if is_right_side else 2
            else:
                correct_quadrant = 4 if is_right_side else 3
        else:
            correct_quadrant = 1 if jaw_type.lower() == "upper" else 4
    
    # Apply corrections
    for i, (idx, ann, num, _) in enumerate(all_teeth):
        position = num % 10
        corrected_fdi = correct_quadrant * 10 + position
        universal_number = fdi_to_universal(corrected_fdi)
        annotations[idx]["original_label"] = str(num)
        annotations[idx]["original_label_after_correction"] = str(corrected_fdi)
        annotations[idx]["model_label"] = annotations[idx]["label"]
        annotations[idx]["label"] = str(universal_number)
    
    return annotations

def process_pano_postprocessing(annotations, img_array):
    """
    Comprehensive postprocessing for panoramic X-ray images including:
    1. Teeth limiting to 32 maximum
    2. Jaw-based tooth positioning
    3. Gap detection and correction
    4. Recursive tooth numbering updates
    """
    
    def calculate_distance(center1, center2):
        return math.sqrt((center2['x'] - center1['x']) ** 2 + (center2['y'] - center1['y']) ** 2)

    def get_tooth_type(tooth_number):
        if tooth_number >= 1 and tooth_number <= 16:
            if tooth_number in [8, 9, 7, 10]:
                return 'incisor'
            if tooth_number in [6, 11]:
                return 'canine'
            if tooth_number in [4, 5, 12, 13]:
                return 'premolar'
            if tooth_number in [1, 2, 3, 14, 15, 16]:
                return 'molar'
        if tooth_number >= 17 and tooth_number <= 32:
            if tooth_number in [24, 25, 23, 26]:
                return 'incisor'
            if tooth_number in [22, 27]:
                return 'canine'
            if tooth_number in [20, 21, 28, 29]:
                return 'premolar'
            if tooth_number in [17, 18, 19, 30, 31, 32]:
                return 'molar'
        return 'unknown'
    
    def determine_jaw_type(tooth, jaw_annotations):
        # First try to find the lower jaw annotation
        lower_jaw = None
        for anno in jaw_annotations:
            if anno.get('label', '').lower() in ['lower jaw', 'mandible', 'lowerjaw']:
                lower_jaw = anno
                break
        
        # If no lower jaw annotation found, use the traditional method
        if lower_jaw is None:
            # Get image height
            image_height = img_array.shape[0]
            # Get tooth center coordinates
            y_center = get_tooth_center(tooth)['y']
            # Return jaw type based on position
            return 'lower' if y_center > (image_height * 0.6) else 'upper'
        
        # Calculate overlap percentage with lower jaw
        if 'segmentation' in tooth and 'segmentation' in lower_jaw:
            overlap_percent = calculate_overlap_percentage(tooth['segmentation'], lower_jaw['segmentation'])
            
            # If overlap is 70% or more, it's a lower jaw tooth
            if overlap_percent >= 70:
                return 'lower'
            else:
                return 'upper'
        else:
            # Fallback to traditional method if segmentation is missing
            image_height = img_array.shape[0]
            y_center = get_tooth_center(tooth)['y']
            return 'lower' if y_center > (image_height * 0.6) else 'upper'

    def limit_teeth_to_32(annotations):
        """
        Limit the number of tooth annotations to exactly 32 by removing the least confident ones.
        Ensures maximum 16 teeth per jaw (upper and lower).
        """
        # Separate tooth annotations from other annotations
        tooth_annotations = []
        other_annotations = []
        
        for anno in annotations:
            if str(anno.get('label', '')).isdigit():
                tooth_annotations.append(anno)
            else:
                other_annotations.append(anno)
        
        # If we have 32 or fewer teeth, return as is
        if len(tooth_annotations) <= 32:
            return annotations
        
        # Filter jaw annotations for determining jaw type
        jaw_annotations = [anno for anno in annotations if anno.get('label', '').lower() in 
                        ['upper jaw', 'maxilla', 'upperjaw', 'lower jaw', 'mandible', 'lowerjaw']]
        
        # Separate teeth by jaw
        upper_jaw_teeth = []
        lower_jaw_teeth = []
        
        for tooth in tooth_annotations:
            jaw_type = determine_jaw_type(tooth, jaw_annotations)
            if jaw_type == 'upper':
                upper_jaw_teeth.append(tooth)
            elif jaw_type == 'lower':
                lower_jaw_teeth.append(tooth)
        
        # Sort each jaw's teeth by confidence (descending order - highest confidence first)
        upper_jaw_teeth.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        lower_jaw_teeth.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Limit to maximum 16 teeth per jaw
        limited_upper_teeth = upper_jaw_teeth[:16]
        limited_lower_teeth = lower_jaw_teeth[:16]
        
        # Combine limited teeth from both jaws
        limited_tooth_annotations = limited_upper_teeth + limited_lower_teeth
        
        # Log the changes
        original_upper_count = len(upper_jaw_teeth)
        original_lower_count = len(lower_jaw_teeth)
        final_upper_count = len(limited_upper_teeth)
        final_lower_count = len(limited_lower_teeth)
        
        print(f"Teeth distribution before limiting: Upper jaw: {original_upper_count}, Lower jaw: {original_lower_count}")
        print(f"Teeth distribution after limiting: Upper jaw: {final_upper_count}, Lower jaw: {final_lower_count}")
        
        if original_upper_count > 16:
            print(f"Removed {original_upper_count - 16} least confident teeth from upper jaw")
        if original_lower_count > 16:
            print(f"Removed {original_lower_count - 16} least confident teeth from lower jaw")
        
        total_removed = (original_upper_count + original_lower_count) - len(limited_tooth_annotations)
        if total_removed > 0:
            print(f"Total teeth removed: {total_removed}")
        
        # Combine limited teeth with other annotations
        return other_annotations + limited_tooth_annotations
    
    def find_unique_molar(annotations):
        # Get image width and height
        image_width = img_array.shape[1]
        image_height = img_array.shape[0]
        
        # Filter jaw annotations
        jaw_annotations = [anno for anno in annotations if anno.get('label', '').lower() in 
                        ['upper jaw', 'maxilla', 'upperjaw', 'lower jaw', 'mandible', 'lowerjaw']]
        
        # Get all teeth with numeric labels
        all_teeth = [anno for anno in annotations if str(anno['label']).isdigit()]
        
        # Separate teeth by quadrant (allowing any valid molar numbers in each quadrant)
        upper_left_molars = []   # Any teeth from [1, 2, 3, 14, 15, 16]
        upper_right_molars = []  # Any teeth from [1, 2, 3, 14, 15, 16]
        lower_left_molars = []   # Any teeth from [17, 18, 19, 30, 31, 32]
        lower_right_molars = []  # Any teeth from [17, 18, 19, 30, 31, 32]
        
        # Define valid molar numbers for each jaw
        upper_molar_numbers = [1, 2, 3, 14, 15, 16]
        lower_molar_numbers = [17, 18, 19, 30, 31, 32]
        
        for tooth in all_teeth:
            tooth_number = int(tooth['label'])
            jaw_type = determine_jaw_type(tooth, jaw_annotations)
            x_center = get_tooth_center(tooth)['x']
            
            if jaw_type == 'upper' and tooth_number in upper_molar_numbers:
                if x_center < (image_width / 2):  # Left side
                    upper_left_molars.append(tooth)
                else:  # Right side
                    upper_right_molars.append(tooth)
            elif jaw_type == 'lower' and tooth_number in lower_molar_numbers:
                if x_center > (image_width / 2):  # Right side
                    lower_right_molars.append(tooth)
                else:  # Left side
                    lower_left_molars.append(tooth)
        
        def correct_molar_positions(molars_list, expected_numbers):
            """
            Correct the positions of 3 molars based on their spatial arrangement
            Returns list of molars with corrected labels
            """
            if len(molars_list) != 3:
                return molars_list
            
            # Sort molars by x-coordinate based on expected numbering pattern
            if expected_numbers == [1, 2, 3]:
                # Upper left: 1 (leftmost) -> 2 -> 3 (rightmost)
                molars_list.sort(key=lambda m: get_tooth_center(m)['x'])
            elif expected_numbers == [14, 15, 16]:
                # Upper right: 14 (leftmost) -> 15 -> 16 (rightmost)  
                molars_list.sort(key=lambda m: get_tooth_center(m)['x'])
            elif expected_numbers == [17, 18, 19]:
                # Lower right: 17 (rightmost) -> 18 -> 19 (leftmost)
                molars_list.sort(key=lambda m: get_tooth_center(m)['x'], reverse=True)
            elif expected_numbers == [30, 31, 32]:
                # Lower left: 30 (rightmost) -> 31 -> 32 (leftmost)
                molars_list.sort(key=lambda m: get_tooth_center(m)['x'], reverse=True)
            
            # Assign correct numbers based on position
            corrected_molars = []
            for i, molar in enumerate(molars_list):
                corrected_molar = molar.copy()
                corrected_molar["original_label"] = molar["label"]
                corrected_molar["label"] = str(expected_numbers[i])
                corrected_molar["created_by"] = "Model v1.0.0 with Auto Labelling - 3 Molar Correction"
                corrected_molar["created_on"] = datetime.now(timezone.utc).isoformat()
                corrected_molar["confidence_in_numbering"] = "high"  # High confidence due to 3-molar detection
                corrected_molars.append(corrected_molar)
            
            return corrected_molars
        
        # Check for complete sets of 3 molars and correct their positioning
        selected_upper_molar = None
        selected_lower_molar = None
        
        # Priority 1: Complete sets of 3 molars (any 3 molars in each quadrant)
        if len(upper_left_molars) == 3:
            corrected_molars = correct_molar_positions(upper_left_molars, [1, 2, 3])
            selected_upper_molar = next(m for m in corrected_molars if m['label'] == '2')  # Use tooth 2 as anchor
            # Update the annotations with corrected molars
            for original, corrected in zip(upper_left_molars, corrected_molars):
                for i, anno in enumerate(annotations):
                    if anno == original:
                        annotations[i] = corrected
            print("Found complete set of 3 upper left molars, corrected their positions to (1,2,3)")
            
        elif len(upper_right_molars) == 3:
            corrected_molars = correct_molar_positions(upper_right_molars, [14, 15, 16])
            selected_upper_molar = next(m for m in corrected_molars if m['label'] == '15')  # Use tooth 15 as anchor
            # Update the annotations with corrected molars
            for original, corrected in zip(upper_right_molars, corrected_molars):
                for i, anno in enumerate(annotations):
                    if anno == original:
                        annotations[i] = corrected
            print("Found complete set of 3 upper right molars, corrected their positions to (14,15,16)")
        
        if len(lower_left_molars) == 3:
            corrected_molars = correct_molar_positions(lower_left_molars, [30, 31, 32])
            selected_lower_molar = next(m for m in corrected_molars if m['label'] == '31')  # Use tooth 31 as anchor
            # Update the annotations with corrected molars
            for original, corrected in zip(lower_left_molars, corrected_molars):
                for i, anno in enumerate(annotations):
                    if anno == original:
                        annotations[i] = corrected
            print("Found complete set of 3 lower left molars, corrected their positions to (30,31,32)")
            
        elif len(lower_right_molars) == 3:
            corrected_molars = correct_molar_positions(lower_right_molars, [17, 18, 19])
            selected_lower_molar = next(m for m in corrected_molars if m['label'] == '18')  # Use tooth 18 as anchor
            # Update the annotations with corrected molars
            for original, corrected in zip(lower_right_molars, corrected_molars):
                for i, anno in enumerate(annotations):
                    if anno == original:
                        annotations[i] = corrected
            print("Found complete set of 3 lower right molars, corrected their positions to (17,18,19)")
        
        # If no complete 3-molar sets found, use original logic
        if selected_upper_molar is None or selected_lower_molar is None:
            print("No complete 3-molar sets found, using original molar selection logic")
            
            # Get all molars using original logic
            molars = [anno for anno in annotations if str(anno['label']).isdigit() and get_tooth_type(int(anno['label'])) == 'molar']
            
            if selected_upper_molar is None:
                for current_molar in molars:
                    tooth_number = int(current_molar['label'])
                    jaw_type = determine_jaw_type(current_molar, jaw_annotations)
                    x_center = get_tooth_center(current_molar)['x']
                    
                    if jaw_type == 'upper':
                        if 1 <= tooth_number <= 3 and x_center < (image_width / 2):
                            selected_upper_molar = current_molar
                            break
                        elif 14 <= tooth_number <= 16 and x_center > (image_width / 2):
                            selected_upper_molar = current_molar
                            break
            
            if selected_lower_molar is None:
                for current_molar in molars:
                    tooth_number = int(current_molar['label'])
                    jaw_type = determine_jaw_type(current_molar, jaw_annotations)
                    x_center = get_tooth_center(current_molar)['x']
                    
                    if jaw_type == 'lower':
                        if 17 <= tooth_number <= 19 and x_center > (image_width / 2):
                            selected_lower_molar = current_molar
                            break
                        elif 30 <= tooth_number <= 32 and x_center < (image_width / 2):
                            selected_lower_molar = current_molar
                            break
            
            # Fallback to other teeth if still no molars found
            if selected_upper_molar is None or selected_lower_molar is None:
                # Create separate collections for each quadrant
                upper_left_fallback = []   # Teeth 4-8
                upper_right_fallback = []  # Teeth 9-13
                lower_right_fallback = []  # Teeth 20-24
                lower_left_fallback = []   # Teeth 25-29
                
                # Get all teeth that aren't already selected
                other_teeth = [anno for anno in annotations if str(anno['label']).isdigit()]
                
                for tooth in other_teeth:
                    tooth_number = int(tooth['label'])
                    x_center = get_tooth_center(tooth)['x']
                    jaw_type = determine_jaw_type(tooth, jaw_annotations)
                    
                    # Upper teeth
                    if jaw_type == 'upper':
                        if 4 <= tooth_number <= 8 and x_center < (image_width / 2):
                            upper_left_fallback.append(tooth)
                        elif 9 <= tooth_number <= 13 and x_center > (image_width / 2):
                            upper_right_fallback.append(tooth)
                    
                    # Lower teeth
                    elif jaw_type == 'lower':
                        if 20 <= tooth_number <= 24 and x_center > (image_width / 2):
                            lower_right_fallback.append(tooth)
                        elif 25 <= tooth_number <= 29 and x_center < (image_width / 2):
                            lower_left_fallback.append(tooth)
                
                # Use fallback if needed for upper teeth
                if selected_upper_molar is None:
                    if upper_left_fallback:
                        selected_upper_molar = upper_left_fallback[0]
                    elif upper_right_fallback:
                        selected_upper_molar = upper_right_fallback[0]
                
                # Use fallback if needed for lower teeth
                if selected_lower_molar is None:
                    if lower_left_fallback:
                        selected_lower_molar = lower_left_fallback[0]
                    elif lower_right_fallback:
                        selected_lower_molar = lower_right_fallback[0]
        
        print("Selected anchor teeth:", 
            "Upper:", selected_upper_molar['label'] if selected_upper_molar else None, 
            "Lower:", selected_lower_molar['label'] if selected_lower_molar else None)
        
        return selected_upper_molar, selected_lower_molar
    
    def recursively_update_teeth_with_gap_detection(
            current_annotations, 
            current_tooth, 
            current_tooth_number, 
            updated_teeth_indexes,
            tooth_width_data,
            jaw_annotations
        ):
        # Get all tooth annotations with numeric labels
        teeth_annotations = [anno for anno in current_annotations if str(anno.get('label')).isdigit()]
        
        # Determine jaw type based on overlap with lower jaw annotation
        jaw_type = determine_jaw_type(current_tooth, jaw_annotations)
        is_upper_jaw = (jaw_type == 'upper')
        is_lower_jaw = (jaw_type == 'lower')
        
        # Find adjacent teeth based on segmentation proximity
        adjacent_teeth = find_adjacent_teeth(current_tooth, teeth_annotations, is_upper_jaw, is_lower_jaw)
        
        updated_annotations = current_annotations.copy()

        # Process the left adjacent tooth
        if adjacent_teeth.get('left'):
            left_tooth = adjacent_teeth['left']
            left_tooth_index = next((index for index, anno in enumerate(current_annotations) if anno == left_tooth), -1)
            if left_tooth_index != -1 and left_tooth_index not in updated_teeth_indexes:
                # Calculate the distance between tooth centers
                distance = calculate_distance(get_tooth_center(current_tooth),get_tooth_center(left_tooth))
                
                # Calculate current tooth width
                current_tooth_width = calculate_tooth_width(current_tooth)
                
                # Calculate left tooth width
                left_tooth_width = calculate_tooth_width(left_tooth)
                
                # Determine what type of tooth would be in the gap
                gap_tooth_type = get_tooth_type(current_tooth_number - 1) if is_upper_jaw else get_tooth_type(current_tooth_number + 1)
                
                # Get the expected width based on tooth type
                expected_width = max(1,tooth_width_data["byType"].get(gap_tooth_type, tooth_width_data["average"]))
                
                # Calculate the gap (distance between edges)
                gap = distance - (current_tooth_width / 2) - (left_tooth_width / 2)
                
                # Calculate how many teeth would fit in this gap
                gap_size = max(1, round(gap / expected_width) + 1)
                
                # Determine the new label with gap consideration
                left_tooth_new_label = (current_tooth_number - gap_size) if is_upper_jaw else (current_tooth_number + gap_size)
                
                # Only update if the new label is valid for the jaw
                if (is_upper_jaw and 1 <= left_tooth_new_label <= 16) or (is_lower_jaw and 17 <= left_tooth_new_label <= 32):
                    updated_left_tooth = left_tooth.copy()
                    updated_left_tooth["label"] = str(left_tooth_new_label)
                    updated_left_tooth["original_label"] = left_tooth["label"]
                    updated_left_tooth["created_by"] = "Model v1.0.0 with Auto Labelling"
                    updated_left_tooth["created_on"] = datetime.now(timezone.utc).isoformat()
                    updated_left_tooth['updated_by'] = current_tooth_number
                    updated_teeth_indexes.add(left_tooth_index)

                    updated_annotations[left_tooth_index] = updated_left_tooth

                    # Recursively update teeth starting from this left tooth
                    updated_annotations = recursively_update_teeth_with_gap_detection(
                        updated_annotations,
                        updated_left_tooth,
                        left_tooth_new_label,
                        updated_teeth_indexes,
                        tooth_width_data,
                        jaw_annotations
                    )

        # Process the right adjacent tooth
        if adjacent_teeth.get('right'):
            right_tooth = adjacent_teeth['right']
            right_tooth_index = next((index for index, anno in enumerate(current_annotations) if anno == right_tooth), -1)

            if right_tooth_index != -1 and right_tooth_index not in updated_teeth_indexes:
                # Calculate the distance between tooth centers
                distance = calculate_distance(get_tooth_center(current_tooth), get_tooth_center(right_tooth))
                
                # Calculate current tooth width
                current_tooth_width = calculate_tooth_width(current_tooth)
                
                # Calculate right tooth width
                right_tooth_width = calculate_tooth_width(right_tooth)
                
                # Determine what type of tooth would be in the gap
                gap_tooth_type = get_tooth_type(current_tooth_number + 1) if is_upper_jaw else get_tooth_type(current_tooth_number - 1)
                
                # Get the expected width based on tooth type
                expected_width = tooth_width_data["byType"].get(gap_tooth_type, tooth_width_data["average"])
                
                # Calculate the gap (distance between edges)
                gap = distance - (current_tooth_width / 2) - (right_tooth_width / 2)
                
                # Calculate how many teeth would fit in this gap
                gap_size = max(1, round(gap / expected_width) + 1)
                
                # Determine the new label with gap consideration
                right_tooth_new_label = (current_tooth_number + gap_size) if is_upper_jaw else (current_tooth_number - gap_size)
                
                # Only update if the new label is valid for the jaw
                if (is_upper_jaw and 1 <= right_tooth_new_label <= 16) or (is_lower_jaw and 17 <= right_tooth_new_label <= 32):
                    updated_right_tooth = right_tooth.copy()
                    updated_right_tooth["label"] = str(right_tooth_new_label)
                    updated_right_tooth["created_by"] = "Model v1.0.0 with Auto Labelling"
                    updated_right_tooth["original_label"] = right_tooth["label"]
                    updated_right_tooth['updated_by'] = current_tooth_number
                    updated_right_tooth["created_on"] = datetime.now(timezone.utc).isoformat()

                    updated_teeth_indexes.add(right_tooth_index)

                    updated_annotations[right_tooth_index] = updated_right_tooth

                    # Recursively update teeth starting from this right tooth
                    updated_annotations = recursively_update_teeth_with_gap_detection(
                        updated_annotations,
                        updated_right_tooth,
                        right_tooth_new_label,
                        updated_teeth_indexes,
                        tooth_width_data,
                        jaw_annotations
                    )
        
        return updated_annotations

    def calculate_average_tooth_width_by_type(annotations):
        # Filter annotations to only include those with valid tooth numbers
        teeth_annotations = [anno for anno in annotations if str(anno.get('label')).isdigit()]
        
        if not teeth_annotations:
            return {'average': 0, 'byType': {}}
        
        # Group teeth by type
        teeth_by_type = {
            'incisor': [],
            'canine': [],
            'premolar': [],
            'molar': [],
            'unknown': []
        }
        
        # Calculate width for each tooth and group by type
        for tooth in teeth_annotations:
            tooth_number = int(tooth['label'])
            tooth_type = get_tooth_type(tooth_number)
            width = calculate_tooth_width(tooth)
            
            # Add to appropriate type group
            if width > 0:
                teeth_by_type[tooth_type].append(width)
        
        # Calculate average width for each type
        averages_by_type = {}
        total_width = 0
        total_count = 0
        
        for tooth_type, widths in teeth_by_type.items():
            if widths:
                sum_widths = sum(widths)
                averages_by_type[tooth_type] = sum_widths / len(widths)
                total_width += sum_widths
                total_count += len(widths)
            else:
                averages_by_type[tooth_type] = 0
        
        # Overall average (fallback)
        overall_average = total_width / total_count if total_count > 0 else 0
        
        return {
            'average': overall_average,
            'byType': averages_by_type
        }

    def find_adjacent_teeth(current_tooth, teeth_annotations, is_upper_jaw, is_lower_jaw):
        left = None
        right = None

        # Get the center of the current tooth
        current_tooth_center = get_tooth_center(current_tooth)
        
        # Calculate the threshold for y-coordinate proximity (15% of image height)
        y_threshold = img_array.shape[0] * 0.15
        
        # Find left adjacent tooth (based on proximity and position)
        min_left_distance = float('inf')
        for tooth in teeth_annotations:
            # Skip if it's the current tooth
            if tooth == current_tooth:
                continue
            
            tooth_center = get_tooth_center(tooth)
            
            # Check vertical proximity (within threshold of the image height)
            y_diff = abs(tooth_center['y'] - current_tooth_center['y'])
            if y_diff > y_threshold:
                continue
                
            distance = calculate_distance(current_tooth_center, tooth_center)
            
            # Check if the tooth is on the left (its center is to the left of the current tooth)
            if tooth_center['x'] < current_tooth_center['x'] and distance < min_left_distance:
                min_left_distance = distance
                left = tooth

        # Find right adjacent tooth (based on proximity and position)
        min_right_distance = float('inf')
        for tooth in teeth_annotations:
            # Skip if it's the current tooth
            if tooth == current_tooth:
                continue
            
            tooth_center = get_tooth_center(tooth)
            
            # Check vertical proximity (within threshold of the image height)
            y_diff = abs(tooth_center['y'] - current_tooth_center['y'])
            if y_diff > y_threshold:
                continue
                
            distance = calculate_distance(current_tooth_center, tooth_center)
            
            # Check if the tooth is on the right (its center is to the right of the current tooth)
            if tooth_center['x'] > current_tooth_center['x'] and distance < min_right_distance:
                min_right_distance = distance
                right = tooth
        
        # Construct and return the result dictionary
        result = {}
        if left is not None:
            result["left"] = left
        if right is not None:
            result["right"] = right
        
        return result

    def get_tooth_center(tooth):
        if tooth['segmentation']:
            # For polygon (segmentation), calculate the centroid
            sum_x = 0
            sum_y = 0
            for point in tooth['segmentation']:
                sum_x += point['x']
                sum_y += point['y']
            return {
                'x': sum_x / len(tooth['segmentation']),
                'y': sum_y / len(tooth['segmentation'])
            }
        return {'x': 0, 'y': 0}

    def calculate_tooth_width(tooth):
        if 'segmentation' in tooth and tooth['segmentation']:
            # For polygon, calculate width based on bounding box of segmentation points
            min_x = min(point['x'] for point in tooth['segmentation'])
            max_x = max(point['x'] for point in tooth['segmentation'])
            return max_x - min_x
        return 0

    # Main postprocessing logic
    # LIMIT TEETH TO 32 BEFORE POST-PROCESSING
    annotations = limit_teeth_to_32(annotations)
    
    # Filter jaw annotations for recursive tooth updating
    jaw_annotations = [anno for anno in annotations if anno.get('label', '').lower() in 
                      ['upper jaw', 'maxilla', 'upperjaw', 'lower jaw', 'mandible', 'lowerjaw']]
    
    # Find unique molar to start the tooth labeling process
    upper_molar, lower_molar = find_unique_molar(annotations)
    
    if upper_molar is not None:
        updated_annotations = recursively_update_teeth_with_gap_detection(
            annotations,
            upper_molar,
            int(upper_molar['label']),
            updated_teeth_indexes=set(),
            tooth_width_data=calculate_average_tooth_width_by_type(annotations),
            jaw_annotations=jaw_annotations
        )
    else:
        updated_annotations = annotations
        
    if lower_molar is not None:
        updated_annotations = recursively_update_teeth_with_gap_detection(
            updated_annotations,
            lower_molar,
            int(lower_molar['label']),
            updated_teeth_indexes=set(),
            tooth_width_data=calculate_average_tooth_width_by_type(annotations),
            jaw_annotations=jaw_annotations
        )
        
    return updated_annotations
