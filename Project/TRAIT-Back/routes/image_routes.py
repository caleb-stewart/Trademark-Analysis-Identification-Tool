from flask import Blueprint, jsonify, request
from flask_cors import CORS

image_blueprint = Blueprint("image", __name__, url_prefix="/image")
CORS(image_blueprint)

@image_blueprint.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Image processing service is running"}), 200

    
@image_blueprint.route("/detect-all", methods=["POST"])
def detect_all():
    from utils.process_image import identify_all_logos

    if "main_image" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    main_image = request.files["main_image"]
    bb_color = request.form.get("bb_color")
    bounding_box_threshold = float(request.form.get("bounding_box_threshold", 0.25))  # with default
    bounding_box_threshold = float(bounding_box_threshold)/100 #convert to decimal percent for func parameter 

    return identify_all_logos(main_image, bb_color,bounding_box_threshold)


@image_blueprint.route("/detect-specific", methods=["POST"])
def detect_specific():
    from utils.process_image import compare_logo_embeddings

    if "main_image" not in request.files or "reference_image" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    main_image = request.files["main_image"]
    reference_image = request.files["reference_image"]
    similarity_threshold = int(request.form.get("confidence"))
    bb_color = request.form.get("bb_color")
    bounding_box_threshold = float(request.form.get("bounding_box_threshold", 0.25))  # with default
    bounding_box_threshold = float(bounding_box_threshold)/100 #convert to decimal percent for func parameter 

    return compare_logo_embeddings(main_image, reference_image, similarity_threshold, bb_color,bounding_box_threshold)

@image_blueprint.route("/cancel", methods=["POST"])
def cancel():
    from utils.cancel_process import cancel_state_image

    cancel_state_image['canceled'] = True
    
    return jsonify({"message": "Processing cancelled"}), 200