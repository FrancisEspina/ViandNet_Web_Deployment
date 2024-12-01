import os
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from skimage import color as skimageColor
from numpy import round as npRound, min as npMin, max as npMax, array as npArray
from PIL import Image
from joblib import load



# Load the model
model = tf.keras.models.load_model("cnn/RGB_XYZ.hdf5")
svm = load("svm/PCNN_FINALNoneColorSpace.XYZ_SVM(RCV).joblib")

CLASS_NAMES = ['Apple Pie', 'Baby Back Ribs', 'Baklava', 'Beef Carpaccio', 'Beef Tartare', 'Beet Salad', 'Beignets', 'Bibimbap', 'Bread Pudding', 'Breakfast Burrito', 'Bruschetta', 'Caesar Salad', 'Cannoli', 'Caprese Salad', 'Carrot Cake', 'Ceviche', 'Cheesecake', 'Cheese Plate', 'Chicken Curry', 'Chicken Quesadilla', 'Chicken Wings', 'Chocolate Cake', 'Chocolate Mousse', 'Churros', 'Clam Chowder', 'Club Sandwich', 'Crab Cakes', 'Creme Brulee', 'Croque Madame', 'Cup Cakes', 'Deviled Eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs Benedict', 'Escargots', 'Falafel', 'Filet Mignon', 'Fish And Chips', 'Foie Gras', 'French Fries', 'French Onion Soup', 'French Toast', 'Fried Calamari', 'Fried Rice', 'Frozen Yogurt', 'Garlic Bread', 'Gnocchi', 'Greek Salad', 'Grilled Cheese Sandwich', 'Grilled Salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot And Sour Soup', 'Hot Dog', 'Huevos Rancheros', 'Hummus', 'Ice Cream', 'Lasagna', 'Lobster Bisque', 'Lobster Roll Sandwich', 'Macaroni And Cheese', 'Macarons', 'Miso Soup', 'Mussels', 'Nachos', 'Omelette', 'Onion Rings', 'Oysters', 'Pad Thai', 'Paella', 'Pancakes', 'Panna Cotta', 'Peking Duck', 'Pho', 'Pizza', 'Pork Chop', 'Poutine', 'Prime Rib', 'Pulled Pork Sandwich', 'Ramen', 'Ravioli', 'Red Velvet Cake', 'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed Salad', 'Shrimp And Grits', 'Spaghetti Bolognese', 'Spaghetti Carbonara', 'Spring Rolls', 'Steak', 'Strawberry Shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna Tartare', 'Waffles']


# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for classification
@app.route('/classify', methods=['POST'])

def classify():
    # Get image data from request
    image = request.files['image']
    # Preprocess image
    image = load_and_prep_image(image)
    # Make prediction by creating a pair
    feature =  model.predict([preproc_custom_test(image, is_rgb=True),preproc_custom_test(image, is_rgb=False)])
    
    predictions = svm.decision_function(feature)
    # Decode predictions
    return jsonify(decode_predictions(predictions))
    
# Function for preparing the image
def load_and_prep_image(image_file, img_shape=224):

    # Read image file
    image_stream = image_file.stream
    # Read image from stream
    img = Image.open(image_stream)
    # Convert image to RGB mode (in case it's not already)
    img = img.convert('RGB')
    # Resize the image
    img = img.resize((img_shape, img_shape))
    # Convert image to numpy array
    img_array = npArray(img)
    # Convert numpy array to tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)

    return img_tensor


# Function for conversion of color space
def preproc_custom_test(proc_img, is_rgb):
    cs = convert_image(proc_img, is_rgb)
    cs = tf.expand_dims(cs, axis=0)
    return cs
    
# Convert color main function
def convert_image(image, is_rgb=True):
    image = to_other_color_space(image, is_rgb)
    image = tf.cast(image, tf.uint8)
    return image
    
#data preprocessing with color space transfrom
def to_other_color_space(image, is_rgb):
    if not is_rgb:
        new_image = get_rgb_2_xyz()(image)
        new_image = rescale_0_to_255(new_image)
    else:
        new_image = image
    
    return new_image

# rescale to 0 to 255 non rgb to process
def rescale_0_to_255(image):
    converted_image = image
    for i in range(3):  # Iterate over RGB channels
        min_val = npMin(converted_image[:,:,i])
        max_val = npMax(converted_image[:,:,i])

        converted_image[:,:,i] = npRound(((converted_image[:,:,i] - min_val) / (max_val - min_val)) * 255)

    return converted_image
    
# Return the RGB2XYZ Method
def get_rgb_2_xyz():
  return getattr(skimageColor, "rgb2xyz")

# Function to decode predictions
def decode_predictions(predictions):
    pred_class = CLASS_NAMES[predictions.argmax()]
    top_5_i = (predictions.argsort())[0][-5:][::-1]
    
    # Adjust probabilities to ensure the lowest level is 0
    min_prob = predictions.min()
    if min_prob < 0:
        predictions -= min_prob
    
    # Rescale probabilities to make the top accuracy relative to the other 4
    top_accuracy = predictions[0][top_5_i[0]]
    if top_accuracy < 1.0:
        predictions[0][top_5_i[0]] = 1.0
        predictions[0][top_5_i[1:]] *= (1 - top_accuracy) / 4
    
    # Ensure all probabilities are non-negative
    predictions = predictions.clip(0)
    
    # Rescale probabilities to sum up to 1
    predictions /= predictions.sum()

    values = (predictions[0][top_5_i] * 10).astype(float) # Multiply by 10
    values /= values.sum()
    print(values[0])
    labels = [CLASS_NAMES[top_5_i[x]] for x in range(5)]
    return {
        'values': values.tolist(),
        'labels': labels,
        'pred_class': pred_class,
        'max_prediction': values[0]
    }

if __name__ == '__main__':
    app.run(debug=True)