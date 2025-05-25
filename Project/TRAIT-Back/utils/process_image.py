from utils.embed import BEiTEmbedding, CLIPEmbedding, ResNetEmbedding

beit = BEiTEmbedding()
clip = CLIPEmbedding()
resnet = ResNetEmbedding()

embedding_models = [beit, clip, resnet]

def compare_logo_embeddings(input_file, reference_file, score_threshold, bb_color,bounding_box_threshold):
    from flask import jsonify
    from PIL import Image

    from utils.cancel_process import cancel_state_image
    from utils.logo_detection_utils import convert_file_to_image, check_if_cancelled, hex_to_bgr, extract_logo_regions, verify_vote, extract_and_record_logo, img_to_base64

    # Convert the input file to an image we can use with OpenCV and YOLO model
    img = convert_file_to_image(input_file)
    # Convert the reference file to an image we can use with OpenCV and YOLO model
    reference_img = convert_file_to_image(reference_file)

    # Convert the color from hex to BGR for OpenCV
    bb_color = hex_to_bgr(bb_color)

    # Make sure the cancel state is set to False
    cancel_state_image['canceled'] = False

    # Get the bounding boxes for the logos in the input image and reference image
    # input_logos is the list of cropped logos
    # input_bboxes is the list of bounding boxes for each logo found in the image (tuple of (x1, y1, x2, y2))
    input_logos, input_bboxes, input_img = extract_logo_regions(img, bounding_box_threshold, return_img=True)
    # reference_logos is the list of cropped logos
    reference_logos, _ = extract_logo_regions(reference_img, bounding_box_threshold)

    if not input_logos or not reference_logos:
        print("No logos detected in one or both images.")
        return jsonify({"error": "No logos detected in one or both images."}), 400

    bounding_boxes_info = []  # Will contain bounding box data

    # Check if the cancel state is set to True
    if check_if_cancelled("image"):
        return jsonify({"message": "Processing cancelled"}), 200
    
    # Creates a dictionary of embeddings for each model
    # ex. input_embeddings = {'BEiTEmbedding': [embedding1, embedding2], 'CLIPEmbedding': [embedding3, embedding4], 'ResNetEmbedding': [embedding5, embedding6]}
    input_embeddings = {type(emb).__name__: [emb.extract_embedding(Image.fromarray(logo)) for logo in input_logos] for emb in embedding_models}
    reference_embeddings = {type(emb).__name__: [emb.extract_embedding(Image.fromarray(logo)) for logo in reference_logos] for emb in embedding_models}

    # Iterate over each input logo and reference logo
    for i in range(len(reference_logos)):
        for j in range(len(input_logos)):
            
            # Gather embeddings per model for this pair
            input_embeds = {emb_model: input_embeddings[emb_model][j] for emb_model in input_embeddings}
            reference_embeds = {emb_model: [reference_embeddings[emb_model][i]] for emb_model in reference_embeddings}

            # If the images are similar enough, draw a bounding box around the logo in the input image
            if verify_vote(input_embeds, reference_embeds, score_threshold, embedding_models):
                bounding_boxes_info.append(extract_and_record_logo(input_img, input_bboxes[j], bb_color))

             # Check if the process has been cancelled
            if check_if_cancelled("image"):
                return jsonify({"message": "Processing cancelled"}), 200


        return jsonify({
            "bounding_boxes": bounding_boxes_info,
            "image": img_to_base64(input_img)
        })
    

def identify_all_logos(file, bb_color, bounding_box_threshold):
    '''Returns the image with bounding boxes around all logos found'''
    import cv2
    from flask import jsonify
    import numpy as np

    from utils.cancel_process import cancel_state_image
    from utils.logo_detection_utils import check_if_cancelled, extract_logo_regions, hex_to_bgr, extract_and_record_logo, img_to_base64

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Make sure the cancel state is set to False
    cancel_state_image['canceled'] = False
    
    # input_logos is the list of cropped logos
    # bounding_boxes is a list of bounding boxes for each logo found in the image (tuple of (x1, y1, x2, y2))
    input_logos, bounding_boxes = extract_logo_regions(img, bounding_box_threshold)

    # Convert for to bgr for OpenCV
    bb_color = hex_to_bgr(bb_color)
    
    # Use this to send bounding box info back to frontend
    bounding_boxes_info = []

    if check_if_cancelled("image"):
        return jsonify({"message": "Processing cancelled"}), 200

    for box in bounding_boxes:
        
        # Save all of the bounding box info into this dict. This is sent to frontend
        bounding_boxes_info.append(extract_and_record_logo(img, box, bb_color))

    
    # Make image base64 so it can be jsonifyed.
    img_base64 = img_to_base64(img)

    return jsonify({
        "bounding_boxes": bounding_boxes_info,
        "image": img_base64
    })