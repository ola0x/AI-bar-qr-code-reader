import io
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app)

ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'png', 'jpeg'}

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
        
    # return jsonify({
    #     "message": "successful",
    #     'result': result
    # })
        
if __name__ == "__main__":
    print("Starting Doc verification...")
    app.run()