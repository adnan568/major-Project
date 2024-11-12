import streamlit as st

st.set_page_config(page_title="Home", layout='wide', page_icon='./images/home.png')
st.title("Object Detection Using YOLO V5")
st.caption('This web application is to demonstrate Object detection')

#Content
st.markdown("""
### This App detects objects from Images
- Automatically detects 20 objects from image 
- [Click here for App](/YOLO_for_image/)

Below give are the object the our model will detect
1. Person
2. Car
3. Chair
4. Bottle
5. Sofa
6. Bicycle
7. Horse
8. Boat
9. Motorbike
10. Cat
11. TV Monitor
12. Cow
13. Sheep
14. Airplane
15. Train
16. Dining Table
17. Bus
18. Potted Plant
19. Bird
20. Dog



            """)