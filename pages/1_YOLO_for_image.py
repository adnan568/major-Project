import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np

yolo = YOLO_Pred(onnx_model='./Models/best.onnx', data_yaml='./Models/data.yaml')

st.set_page_config(page_title="YOLO Object Detection", layout='wide', page_icon='./images/object.png')
st.title("Welcome to YOLO for images")
st.write('Please upload image for detections')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./Models/best.onnx', data_yaml='./Models/data.yaml')
    #st.balloons()
 
def upload_image():   
    #Upload Image
    Image_file = st.file_uploader(label='upload Image')
    if Image_file is not None:
        size_mb= Image_file.size/(1024**2)
        file_details ={'filename':Image_file.name, 'filetype':Image_file.type, 'filesize':"{:,.2f} MB".format(size_mb)}
        #st.json(file_details)
        #validate file
        if file_details['filetype'] in ('image/png', 'image/jpeg','image/jpg' ):
            st.success('VALID IMAGE file type(png or jpeg/jpg)')
            return {"file": Image_file,
                    "details": file_details}
        
        
        else:
            st.error('INVALID image type')
            st.error('upload only png, jpeg or jpg format')
            return None
        
def main():
    object= upload_image()
    
    if object:
        prediction = False
        image_obj = Image.open(object['file'])        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)
            
        with col2:
            st.subheader('check below for file details')
            st.json(object['details'])
            button = st.button('Get Detection from YOLO')
            if button:
                with st.spinner("""
                Getting Objects from image. Please wait
                                
                                """):
                    image_array = np.array(image_obj)
                    pred_image = yolo.predictions(image_array)
                    pred_image_obj =Image.fromarray(pred_image)
                    prediction = True
                
        if prediction:
            st.subheader("Predictions:")
            st.image(pred_image_obj)
                
                
    
    
if __name__== "__main__":
    main()