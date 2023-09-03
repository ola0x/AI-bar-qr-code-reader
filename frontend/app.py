import cv2
import numpy as np
import streamlit as st

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from pyzbar.pyzbar import decode as qr_decode

metadataq = MetadataCatalog.get("__")
metadataq.thing_classes = ['barcode-qrcode-1205-combined-110', 'bar_code', 'qr_code']


@st.cache_resource
def initialization():
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85  
    cfg.MODEL.WEIGHTS = "model_final.pth" 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    predictor = DefaultPredictor(cfg)

    return cfg, predictor

def inference(predictor, img):
    return predictor(img)

def output_image(metadata, img, outputs):
    v = Visualizer(img[:, :, ::-1], metadata=metadataq, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()

    return processed_img

def decoder(image):
    decoded_data = None
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    decoded_qr_codes = qr_decode(gray_img)
    
    if decoded_qr_codes:
        qr = decoded_qr_codes[0]
        qrCodeData = qr.data.decode("utf-8")
        decoded_data = qrCodeData
    
    return decoded_data

st.cache_resource
def scanner(image, output):
    instances = output["instances"]
    for i in range(len(instances)):  # Loop through each detected instance
        instance = instances[i]
        box = instance.pred_boxes.tensor[0].cpu().numpy()  # Get the bounding box coordinates
        x1, y1, x2, y2 = box
        cropped_object = image[int(y1):int(y2), int(x1):int(x2)]
        cropped_object = cropped_object[:, :, ::-1]  
        st.image(cropped_object) 

        st.text(decoder(cropped_object)) 

def main():
    # Initialization
    cfg, predictor = initialization()

    # Streamlit initialization
    html_temp = """
        <div style="background-color:black;padding:5px">
        <h2 style="color:white;text-align:center;">Bar Codes/ QR Codes scanner and reader</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Retrieve image
    uploaded_img = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Detection code
        outputs = inference(predictor, img)
        out_image = output_image(cfg, img, outputs)
        st.image(out_image, caption='Processed Image', use_column_width=True)

        scanner(img, outputs)    


if __name__ == '__main__':
    main()