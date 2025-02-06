from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the Python Web Service!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming JSON data is sent in the request body
    # Assuming "City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals" are available in data
    input_data = [data["City"], data["Season"], data["Month"], data["Promotions"], data["Offers"], data["ShippingOptions"], data["LocalFestivals"]]
    input_data = np.array(input_data).reshape(1, -1)  # Reshape input data for prediction

    # Load the trained model
    restored_model = tf.keras.models.load_model('lp_model.keras')

    # Perform prediction using the loaded model
    y_pred = restored_model.predict(input_data)

    # Assuming y_pred contains the predicted class index
    predicted_class_index = int(np.round(y_pred[0, 0]))

    # Assuming CATEGORIES contains the class labels
    CATEGORIES = [
"AliceBlue","AntiqueWhite","Aqua","Aquamarine","Azure","Beige","Bisque","Black","BlanchedAlmond","Blue","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","DarkOrange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Green","GreenYellow","HoneyDew","HotPink","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Magenta","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","Olive","OliveDrab","Orange","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","Purple","Red","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","Yellow","YellowGreen","RebeccaPurple","LightSalmon","Salmon","DarkSalmon","LightCoral","IndianRed","Crimson","FireBrick","DarkRed","Red","Pink","LightPink","HotPink","DeepPink","MediumVioletRed","PaleVioletRed","Coral","Tomato","OrangeRed","DarkOrange","Orange","Gold","Yellow","LightYellow","LemonChiffon","LightGoldenRodYellow","PapayaWhip","Moccasin","PeachPuff","PaleGoldenRod","Khaki","DarkKhaki","Lavender","Thistle","Plum","Violet","Orchid","Fuchsia","Magenta","MediumOrchid","MediumPurple","RebeccaPurple","BlueViolet","DarkViolet","DarkOrchid","DarkMagenta","Purple","Indigo","SlateBlue","DarkSlateBlue","MediumSlateBlue","GreenYellow","Chartreuse","LawnGreen","Lime","SpringGreen","MediumSpringGreen","LightGreen","PaleGreen","DarkSeaGreen","MediumSeaGreen","SeaGreen","ForestGreen","Green","DarkGreen","YellowGreen","OliveDrab","Olive","DarkOliveGreen","MediumAquaMarine","DarkCyan","Teal","Aqua","Cyan","LightCyan","PaleTurquoise","AquaMarine"]
  # Replace with your actual class labels
    predicted_class = CATEGORIES[predicted_class_index]

    prediction = [{'predicted_web_theme_color_base': predicted_class}, {"tensorVersion": tf.__version__}]
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
