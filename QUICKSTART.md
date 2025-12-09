# Quick Start Guide

## Running the Application

1. Open a terminal/command prompt in this folder
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and go to: http://localhost:5000

## Testing with Sample Data

A sample Iris dataset (`sample_iris_dataset.csv`) is included for testing:
- 4 features: sepal_length, sepal_width, petal_length, petal_width
- Target column: species (0, 1, or 2)
- 70 samples

### Steps to test:
1. Upload `sample_iris_dataset.csv`
2. Select "species" as target column
3. Choose any preprocessing method
4. Use 80-20 split
5. Train either Logistic Regression or Decision Tree
6. View your results!

## Troubleshooting

**If you see "Module not found":**
```
pip install -r requirements.txt --upgrade
```

**If port 5000 is in use:**
Edit app.py and change the last line to use a different port:
```python
app.run(debug=True, port=5001)
```

**If you see connection errors:**
Make sure the Flask server is running (you should see output in the terminal)

Enjoy building your ML pipelines! 🚀
