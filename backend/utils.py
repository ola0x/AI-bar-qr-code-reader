import cv2
import io
import fitz
from PIL import Image
from pyzbar.pyzbar import decode as qr_decode

def decoder(image):
    decoded_data = None
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    decoded_qr_codes = qr_decode(gray_img)
    
    if decoded_qr_codes:
        qr = decoded_qr_codes[0]
        qrCodeData = qr.data.decode("utf-8")
        decoded_data = qrCodeData
    
    return decoded_data

def process_doc(file_stream):
    file_extension = file_stream.filename.split('.')[-1].lower()

    if file_extension in ['jpg', 'jpeg', 'png']:
        image_data = file_stream.read()
        return [image_data]  # Wrap the single image in a list
    elif file_extension == 'pdf':
        pdf_data = file_stream.read()
        pdf_document = fitz.open(stream=pdf_data, filetype='pdf')
        images = []
        for page_number in range(pdf_document.page_count):
            pdf_page = pdf_document[page_number]
            image_list = pdf_page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                image = pdf_document.extract_image(xref)
                image_data = image["image"]
                image_extension = image["ext"]
                images.append(image_data)
        return images
    else:
        return None