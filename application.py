from fastai.vision.all import *
import streamlit as st
import plotly.express as px


st.title('Animal Classification Model')
cam_pic = st.camera_input("Take a picture")
#Rasmnni yuklash
file = st.file_uploader('Rasm yukash', type = ['jpeg', 'png', 'gif', 'svg', 'webp'])
if file:
    img  = PILImage.create(file)
    st.image(file)

    #Upload model
    model = load_learner('animal_model.pkl')

    #make a prediction
    model.predict(img)

    #cheking prediction
    pred, pred_id, probs = model.predict(img)

    #Show the prediction result on the display
    st.success(f'Bashorat : {pred}')
    st.info(f'Bashorat : {probs[pred_id]*100:.1f}%')
    
    #plotting
    fig = px.bar(x=probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)
if cam_pic:
    img  = PILImage.create(cam_pic)
    st.image(cam_pic)

    #Upload model
    model = load_learner('animal_model.pkl')

    #make a prediction
    model.predict(cam_pic)

    #cheking prediction
    pred1, pred_id1, probs1 = model.predict(cam_pic)

    #Show the prediction result on the display
    st.success(f'Bashorat : {pred1}')
    st.info(f'Bashorat : {probs1[pred_id1]*100:.1f}%')
    
    #plotting
    fig1 = px.bar(x=probs1*100, y = model.dls.vocab)
    st.plotly_chart(fig1)
