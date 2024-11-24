# Mobile Price Classifier

A machine learning project designed to classify mobile phones into predefined price ranges based on their technical specifications. This project utilizes **Logistic Regression** and **Decision Trees** as core algorithms to develop predictive models, offering insights into how various features influence pricing.

## Project Goals

- **Data Analysis**: Understand the relationships between mobile phone specifications and price ranges.
- **Model Building**:
  - Develop a **Logistic Regression** model for classification.
  - Compare its performance against a **Decision Tree** model.
- **Evaluation**: Assess and compare model performance using key metrics.

## Features

- Comprehensive **Exploratory Data Analysis (EDA)** to uncover trends and insights.
- Implementation of two machine learning models for classification:
  - **Logistic Regression**: A linear approach for classification.
  - **Decision Tree**: A rule-based non-linear model.
- **Performance Evaluation**:
  - Use metrics like accuracy, precision, recall, and F1-score to compare models.
- Visual representation of results for better interpretability.

## Dataset

The dataset used in this project is sourced from the [Mobile Price Range Prediction](https://www.kaggle.com/mbsoroush/mobile-price-range) competition on Kaggle. It contains features representing mobile phone specifications and their corresponding price range categories.

## Requirements

- **Python 3.8+**
- Libraries:
  - `pandas` and `numpy` for data manipulation.
  - `matplotlib` and `seaborn` for data visualization.
  - `scikit-learn` for machine learning model development and evaluation.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mobile-price-classifier.git
   cd mobile-price-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

4. Follow the instructions in the notebook to preprocess the data, train models, and evaluate their performance.

## Evaluation Metrics

The project evaluates the models using the following metrics:
- **Accuracy**: The proportion of correctly classified instances.
- **Precision**: The ratio of correctly predicted positive observations to total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all actual positives.
- **F1-Score**: The harmonic mean of precision and recall.

## Visualizations

- Heatmaps and pair plots to explore data correlations.
- Model performance comparison through bar charts and classification reports.

## Contributions

Contributions are welcome! If you encounter any issues or have suggestions for improvement, feel free to open an **issue** or submit a **pull request**.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
