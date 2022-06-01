from fastai.vision.all import *
import streamlit as st
import plotly.express as px


st.title('Animal Classification Model')
st.text('Hayvonlarni farqlab beruvchi model'\n
       'by me')
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
