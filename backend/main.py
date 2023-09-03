import io
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from utils import process_doc
from model.model_utils import initialization, inference, output_image
from backend.utils import decoder

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app)

ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'png', 'jpeg'}

def scanner(image, output):
    instances = output["instances"]
    for i in range(len(instances)):  # Loop through each detected instance
        instance = instances[i]
        box = instance.pred_boxes.tensor[0].cpu().numpy()  # Get the bounding box coordinates
        x1, y1, x2, y2 = box
        cropped_object = image[int(y1):int(y2), int(x1):int(x2)]
        cropped_object = cropped_object[:, :, ::-1]   

        return decoder(cropped_object)

@app.route('/healthz', methods = ['GET'])
def health():
    return jsonify(
      application='API working',
      version='1.0.0',
      message= "endpoint working..."
    )

@app.route('/read_doc', methods=['POST'])
def verify_document():
    
    if 'document' not in request.files:
        return jsonify({'error': 'No document file part'})

    document = request.files['document']

    uploaded_images = process_doc(document)

    if uploaded_images is None:
        return jsonify({'error': 'Unsupported file format'})
    
    result = []
    
    for img1_ in uploaded_images:
        outputs = inference(predictor, img1_)

        result.append(scanner(img1_, outputs))
        
    return jsonify({
        "message": "successful",
        'result': result
    })
        
if __name__ == "__main__":
    print("Starting Doc verification...")
    app.run()

    cfg, predictor = initialization()