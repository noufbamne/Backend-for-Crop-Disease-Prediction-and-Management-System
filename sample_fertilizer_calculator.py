from flask import Flask, request, render_template

app = Flask(__name__)

# Sample fertilizer data (per tree)
CROP_DATA = {
    "Mango": {"N": 0.5, "P": 0.25, "K": 0.25},  # kg per tree
    "Guava": {"N": 0.2, "P": 0.1, "K": 0.1},
    "Banana": {"N": 0.3, "P": 0.2, "K": 0.15}
}

# Nutrient percentages in fertilizers
FERTILIZERS = {
    "Urea": {"N": 46},
    "DAP": {"N": 18, "P": 46},
    "MOP": {"K": 60}
}


def calculate_fertilizer(crop, num_trees):
    # Get nutrient requirements for the crop
    crop_data = CROP_DATA.get(crop)
    if not crop_data:
        return {"error": "Crop not supported"}
# Calculate total nutrient needs
    total_nutrients = {key: value * num_trees for key, value in crop_data.items()}

    # Calculate fertilizer requirements
    fertilizer_amounts = {}
    for fertilizer, nutrients in FERTILIZERS.items():
        amount = 0
        for nutrient, percentage in nutrients.items():
            if nutrient in total_nutrients and total_nutrients[nutrient] > 0:
                amount += total_nutrients[nutrient] / (percentage / 100)
                total_nutrients[nutrient] = 0  # Reset the nutrient once satisfied
        fertilizer_amounts[fertilizer] = round(amount, 2)

    return fertilizer_amounts


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    crop = request.form.get('crop')
    num_trees = int(request.form.get('num_trees', 0))
    if num_trees <= 0:
     return {"error": "Number of trees must be greater than zero"}

    result = calculate_fertilizer(crop, num_trees)
    return render_template('result.html', crop=crop, num_trees=num_trees, result=result)


if __name__ == '__main__':
    app.run(debug=True)