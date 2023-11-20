import firebase_admin
from firebase_admin import credentials
import firebase_admin
from firebase_admin import ml
from firebase_admin import credentials
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
source = ml.TFLiteGCSModelSource.from_tflite_model_file('cats_model.tflite')

# Create the model object
tflite_format = ml.TFLiteFormat(model_source=source)
model = ml.Model(
    display_name="cats_model",  # This is the name you use from your app to load the model.
    tags=["examples"],             # Optional tags for easier management.
    model_format=tflite_format)

# Add the model to your Firebase project and publish it
new_model = ml.create_model(model)
ml.publish_model(new_model.model_id)
