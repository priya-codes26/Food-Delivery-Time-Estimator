# Food Delivery Time Estimator using Machine Learning

## Included files
- `dataset/food_delivery.csv`: 2,000 illustrative synthetic records.
- `train_model.py`: trains a Random Forest regression pipeline and saves it in `model/`.
- `app.py`: Streamlit dashboard with prediction form, prediction history and analytics.
- `requirements.txt`: Python packages.

## Run in VS Code (Windows)

Open the project folder in VS Code, then open Terminal.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The dashboard normally opens at http://localhost:8501.

## Important dataset note
The included CSV is synthetic, generated for development and demonstration. It is not collected from a food delivery company and should not be described as real-world data. For a final major-project evaluation, document the dataset source, clean a suitable real dataset if available, retrain the model, and report test-set metrics honestly.
