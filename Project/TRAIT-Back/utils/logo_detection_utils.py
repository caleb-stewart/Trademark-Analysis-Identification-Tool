def hex_to_bgr(hex_color):
    '''Converts a hex color to bgr (blue, green, red)
    We do it this way because thats how OpenCV reads colors'''
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr


def compute_cosine_similarity(embedding1, embedding2):
    from scipy.spatial.distance import cosine
    return 1 - cosine(embedding1.ravel(), embedding2.ravel())


def compute_euclidean_distances(embedding1, embedding2):
    from scipy.spatial.distance import euclidean
    return euclidean(embedding1.ravel(), embedding2.ravel())


def extract_logo_regions(image, bounding_box_threshold, save_crop=False, output_dir="cropped_logos", return_img=False):
    """Runs YOLO on an image and extracts detected logo regions."""

    import cv2
    import os

    from models.model_load import initialize_model
    model = initialize_model()

    # Check if input is a file path or an image array
    if isinstance(image, str):  # File path
        img = cv2.imread(image)
    else:  # Assume it's already an ndarray
        img = image

    if img is None:
        print("Error: Could not load image.")
        return [], [], None
    print("Boundary box threshold:", bounding_box_threshold)
    results = model(img, conf=bounding_box_threshold, iou=0.5)
    

    # A list of cropped logos
    logo_regions = []
    # A list of bounding boxes
    bounding_boxes = []

    if save_crop and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, box in enumerate(results[0].boxes):
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        cropped_logo = img[y1:y2, x1:x2]  # extract detected region

        if save_crop and cropped_logo.size > 0:
            cropped_logo_path = os.path.join(output_dir, f"cropped_logo_{idx}.jpg")
            cv2.imwrite(cropped_logo_path, cropped_logo)
            print(f"Logo {idx} saved: {cropped_logo_path}")

        if cropped_logo.size > 0:
            logo_regions.append(cropped_logo)
            bounding_boxes.append((x1, y1, x2, y2))
            print(f"Logo {idx} detected at coordinates: ({x1}, {y1}) -> ({x2}, {y2})")

    
    if return_img:
        return logo_regions, bounding_boxes, img
    else:
        return logo_regions, bounding_boxes


def img_to_base64(img):
    '''Converts an image (ndarray) to a base64-encoded string
    This is so we can send the image back to the frontend using a json object'''

    import base64
    import cv2

    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    return base64.b64encode(img_bytes).decode('utf-8')


def draw_bb_box(bbox, frame, ID, bb_color=(255, 255, 255)):
    '''
    Draws the bounding box around the logo in the frame
    
    bbox is the coordinates of the bouning box
    frame is the image we want to draw on
    ID is the identification of the detected logo
    '''

    import cv2

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), bb_color, 5)
    cv2.putText(frame, f"ID: {ID}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bb_color, 2)


def add_logo_to_faiss(faiss_index, embedding, logo_id_map, logo_id_counter):
    '''
    Add a new logo embedding to the FAISS index and update the ID map
    Assumes the the embedduing is already normalized
    '''
    # Add the embedding to the FAISS index (already normalized)
    faiss_index.add(embedding)
    # Update the logo ID map with the new logo ID
    logo_id_map[faiss_index.ntotal - 1] = logo_id_counter
    # Increment the logo ID counter for the next unique logo
    return logo_id_counter + 1


def increase_logo_appearance_count(logo_appearance_counts, logo_id_map, I, id):
    """ Increases the count of appearances for a logo in the logo_appearance_counts dictionary"""
    # Get the ID of the logo from the FAISS index
    id = logo_id_map[I[0][0]]
    # Increase the count of appearances for this logo
    logo_appearance_counts[id] += 1
    # Dont save the frame. We've already seen this logo
    save_frame = False
    return id, save_frame


def update_logo_in_faiss(faiss_index, embedding, logo_id_map, logo_appearance_counts, threshold=0.5):
    '''Search FAISS index for a logo and update counts if a match is found or a new entry is added'''

    import numpy as np

    save_frame = False
    # Get the distance and index of the nearest neighbor
    D, I = faiss_index.search(np.array(embedding), k=1)

    # Always add the embedding to FAISS (whether it's a duplicate or new)
    faiss_index.add(embedding)
    # Index of the newly added embedding
    new_faiss_index = faiss_index.ntotal - 1  
    
    # Check if the distance is less than the threshold (LOWER IS BETTER)
    if D[0][0] < threshold:
        # Logo already exists - get the logo ID from the matched embedding
        matched_faiss_index = I[0][0]
        matched_logo_id = logo_id_map[matched_faiss_index]
        
        # Map the new FAISS index to the same logo ID as the matched one
        logo_id_map[new_faiss_index] = matched_logo_id
        
        # Increase the count of appearances for this logo ID
        logo_appearance_counts[matched_logo_id] += 1
        
        assigned_id = matched_logo_id
        
        # We've seen this logo before. Do not save the new frame
        save_frame = False  
        
    else:
        # New logo seen - create a new logo ID
        new_logo_id = len(set(logo_id_map.values()))  # Get next unique logo ID
        
        # Map the new FAISS index to the new logo ID
        logo_id_map[new_faiss_index] = new_logo_id
        
        # Set the number of times we've seen this new logo to 1 (first appearance)
        logo_appearance_counts[new_logo_id] = 1
        
        assigned_id = new_logo_id
        save_frame = True

    # Return the assigned logo ID and whether to save the frame
    return assigned_id, save_frame


def verify_vote(input_embeddings, reference_embeddings, votes_needed, embedding_models):
    ''' 
    Parameters needed:
    input_embeddings (dict)
    reference_embeddings (dict)
    votes_needed (int)
    
    Returns a True or False. If the number if votes is great enough
    
    '''
    thresholds = {
        'BEiTEmbedding': {'cosine': .3, 'euclidean': 110},
        'CLIPEmbedding': {'cosine': .65, 'euclidean': 7.5},
        'ResNetEmbedding': {'cosine': .75, 'euclidean': 50}
        }
    votes = 0
    
    for emb_model in embedding_models:
        emb_model_name = type(emb_model).__name__
        for ref_embedding in reference_embeddings[emb_model_name]:
            cosine_sim = compute_cosine_similarity(input_embeddings[emb_model_name], ref_embedding)
            euclidean_dist = compute_euclidean_distances(input_embeddings[emb_model_name], ref_embedding)
    
            if cosine_sim >= thresholds[emb_model_name]['cosine']:
                votes += 1
            if euclidean_dist <= thresholds[emb_model_name]['euclidean']:
                votes += 1

            # If we reach the number of votes needed, return True.
            # Has a chance to return early, saving computation time
            if votes >= votes_needed:
                return True

    print(f"Votes: {votes}")
    # Number of votes needed never reached. Return False
    return False


def extract_and_record_logo(image, bbox, bb_color):
    '''
    Draws a bounding box around the logo and returns the bounding box info
    
    Expects:
    - image: the image to draw on
    - bbox: the bounding box coordinates (x1, y1, x2, y2)
    - bb_color: the color of the bounding box in brg format
    '''

    import cv2

    # unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Draw the bounding box on the image using the correct color. Thickness of 2
    cv2.rectangle(image, (x1, y1), (x2, y2), bb_color, 2)


    # Calculate the dimensions and area of the bounding box
    box_width = round(x2 - x1, 2)
    box_height = round(y2 - y1, 2)
    box_area = round(box_width * box_height, 2)

    # Get the dimensions of the image and calculate the coverage percentage
    image_height, image_width = image.shape[:2]
    total_image_area = image_width * image_height
    # How much the bounding box covers the image
    coverage_percentage = round((box_area / total_image_area) * 100, 2)

    box_info = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "box_width": box_width,
        "box_height": box_height,
        "box_area": box_area,
        "box_coverage_percentage": coverage_percentage,
        "cropped_logo": img_to_base64(image[y1:y2, x1:x2])
    }

    # Return a dict with the bounding box info
    return box_info


def convert_file_to_image(file):
    '''Converts a file to an image so OpenCV and YOLO model can use it'''

    import cv2
    import numpy as np

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def save_frame_func(frame, frame_idx, logo_id_counter, input_logo, save_dir="new_logo_frames"):
    ''' Saves the frame with the logo bounding box and returns the logo ID and base64 encoded logo '''

    import base64
    import cv2

    # Convert the logo to base64 to be sent to the frontend
    _, buffer = cv2.imencode('.jpg', input_logo)
    logo_b64 = base64.b64encode(buffer).decode('utf-8')

    # Return the frame index, logo ID, and base64 encoded logo
    # This will be appended to the saved_frame_data list, and then sent to frontend
    return {
        "frame_idx": frame_idx,
        "logo_id": logo_id_counter,
        "logo_base64": logo_b64
    }


def check_if_cancelled(media_type):
    '''Check if the process has been cancelled'''

    from utils.cancel_process import cancel_state_image
    from utils.cancel_process import cancel_state_video

    if media_type == "image":
        # Check if the cancel_process flag is set to True
        if cancel_state_image['canceled'] == True:
            print("PROCESS CANCELLED")
            # Reset the cancel_process flag for next use
            cancel_state_image['canceled'] = False
            return True
    
    elif media_type == "video":
        if cancel_state_video['canceled'] == True:
            print("PROCESS CANCELLED")
            # Reset the cancel_process flag for next use
            cancel_state_video['canceled'] = False
            return True
    
    # Process was not cancelled
    return False