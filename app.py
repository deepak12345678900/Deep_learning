import streamlit as st
import openai
import numpy as np
openai.organization = "org-T8IGZCF6EmGEh8fDpMKtVdYI"
openai.api_key = "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"
from PIL import Image


import tensorflow as tf
import pandas as pd
st.title("Image Classifier")
items=['Neem','Aloe vera','Ashwagandha','Thulasi']

df_dict ={
    'Plant_name':['Neem','Aloe Vera','Ashwagandha','Thulasi'],
    'Uses':['''Neem, scientifically known as Azadirachta indica, is native to the Indian subcontinent. Its origin can be traced back to India, Pakistan, Bangladesh, and neighboring countries in South Asia. Neem has been a part of the region's traditional medicine, agriculture, and culture for thousands of years.The neem tree is well adapted to the arid and semi-arid regions of South Asia and has been cultivated and used by local communities for its various beneficial properties. Over time, the knowledge of neem and its uses has spread to other parts of the world, and neem is now grown and utilized in many countries for its medicinal, agricultural, and industrial applications.''','Great','Best',

'''Thulasi, also known as Holy Basil (Ocimum sanctum), has various uses, including:
Religious and spiritual significance in Hindu culture.
Medicinal applications in Ayurvedic medicine for respiratory, digestive, and stress-related issues.
Culinary uses to flavor dishes, make herbal teas, and prepare sweets.
Potential mosquito-repellent properties due to its scent.
Use in hair care and dental hygiene.
Application on insect bites for relief from itching and discomfort.''']}
df_dict['Uses'] = [use.replace('\n', ' ') for use in df_dict['Uses']]

df=pd.DataFrame(df_dict,columns=['Plant_name','Uses'])

uploaded_file = st.file_uploader("Select an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    predict_button = st.button("Predict Image")
    if predict_button:
        img = Image.open(uploaded_file)
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        model = tf.keras.models.load_model('MobileNetV2Model.h5',compile=False)  # Load the model
        predictions = model.predict(np.expand_dims(img, axis=0))
        predicted_class_index = np.argmax(predictions)
        predicted_plant = items[predicted_class_index]
        plant_name=df.iloc[predicted_class_index]['Plant_name']
        # Uses = df.iloc[predicted_class_index]['Uses'].rstrip().split('.')
        # st.header(f'Predicted plant name is {plant_name} ')
        # st.write('Gathering required details of the plants...')
        # st.write(f'Displaying the uses of the plant {plant_name}')
        # st.header("Uses: ")
        # for k in Uses:
        #     if k=='':
        #         continue

        #     st.write('*',k)
        st.title(f'Predicted plant name is : {plant_name} ')
        messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]


        message = f"Describe {plant_name} uses and their Geographical locations in india"
        if message:
            messages.append(
            {"role": "user", "content": message},
        )
            chat = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", messages=messages
		)
        reply = chat.choices[0].message.content
        st.title("Descriptions:")
        st.write(reply)


            

        