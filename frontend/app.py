import cv2
import numpy as np
import streamlit as st
from model.model_utils import initialization, inference, output_image
from backend.utils import decoder

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