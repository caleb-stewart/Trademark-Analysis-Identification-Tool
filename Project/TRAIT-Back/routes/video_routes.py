from flask import Blueprint, request, send_file, jsonify
from flask_cors import CORS

video_blueprint = Blueprint("video", __name__, url_prefix="/video")
CORS(video_blueprint)

@video_blueprint.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Video processing service is running"}), 200

@video_blueprint.route("/detect-all", methods=["POST"])
def detect_all():
    import io
    import os
    import tempfile

    from utils.process_video import process_video

    if "main_video" not in request.files:
        return {"error": "No file provided"}, 400
    
    main_video = request.files["main_video"]
    bb_color = request.form.get("bb_color")
    bounding_box_threshold = float(request.form.get("bounding_box_threshold", 0.25))
    bounding_box_threshold = float(bounding_box_threshold)/100 #convert to decimal percent for func parameter 
    video_stream = io.BytesIO(main_video.read())
    

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_stream.read())
        temp_video_path = tmp_file.name
    
    response = process_video(temp_video_path,bounding_box_threshold, bb_color)

    os.remove(temp_video_path)

    return response

@video_blueprint.route("/fetch-processed-video", methods=["GET"])
def fetch_processed_video():
    import os
    import tempfile

    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "processed_video.mp4")

    if not os.path.exists(video_path):
        return jsonify({
            "error": "Video not found",
            "tried_path": video_path
        }), 404

    return send_file(video_path, mimetype='video/mp4')


@video_blueprint.route("/fetch-progress", methods=["GET"])
def fetch_progress():
    from utils.video_progress import video_progress

    # Return the current progress of the video processing
    # print("*****************************")
    # print(video_progress)
    return video_progress

@video_blueprint.route("/detect-specific", methods=["POST"])
def detect_specific():
    import io
    import os
    import tempfile

    from utils.process_video import process_video_specific;

    if "main_video" not in request.files or "reference_image" not in request.files:
        return {"error": "No file provided"}, 400
    
    main_video = request.files["main_video"]
    reference_image = request.files["reference_image"]
    similarity_threshold = int(request.form.get("confidence"))
    bb_color = request.form.get("bb_color")
    bounding_box_threshold = float(request.form.get("bounding_box_threshold", 0.25))  # with default
    bounding_box_threshold = float(bounding_box_threshold) / 100 #convert to decimal percent for func parameter 


    # Save the main video to a temporary file
    video_stream = io.BytesIO(main_video.read())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
        tmp_video_file.write(video_stream.read())
        temp_video_path = tmp_video_file.name

    # Save the reference image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img_file:
        tmp_img_file.write(reference_image.read())
        temp_img_path = tmp_img_file.name

    response = process_video_specific(temp_video_path, temp_img_path, bounding_box_threshold, bb_color, similarity_threshold)

    os.remove(temp_video_path)
    os.remove(temp_img_path)

    return response

@video_blueprint.route("/cancel", methods=["POST"])
def cancel():
    from utils.cancel_process import cancel_state_video
    # Set the cancel_process flag to True
    # This will set a trigger within the video processing function to return early
    cancel_state_video['canceled'] = True
    return {"message": "Processing cancelled"}