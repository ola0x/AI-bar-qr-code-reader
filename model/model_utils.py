import streamlit as st
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

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