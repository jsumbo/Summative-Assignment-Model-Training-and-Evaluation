# Student Employability Prediction: A Machine Learning Approach

## 📋 Project Overview

This project implements a machine learning pipeline to predict high school student employability using both traditional machine learning and deep learning approaches. The project uses the UCI Student Performance Dataset to develop and compare multiple algorithms for identifying factors that contribute to student career readiness.

## 🎯 Key Features

- **Real Dataset**: UCI Student Performance Dataset (649 students, 33 attributes)
- **Multiple Approaches**: Traditional ML (Logistic Regression, Random Forest, SVM) + Deep Learning (Sequential & Functional APIs)
- **7+ Experiments**: Systematic hyperparameter tuning and model comparison
- **Comprehensive Analysis**: Feature importance, error analysis, and performance evaluation
- **Production Ready**: Well-documented, modular code following SOLID principles


### Running the Notebook
1. **Clone/Download** this repository
2. **Open** `student_employability_prediction.ipynb` in Jupyter Lab or Google Colab
3. **Run all cells** to execute the complete pipeline
4. **View results** in generated visualizations and CSV files

## 🔬 Methodology

### Data Preprocessing
- **Missing value handling**: Dataset had no missing values
- **Categorical encoding**: Label encoding for categorical variables
- **Feature engineering**: Composite scores for academic performance, family support, engagement
- **Data scaling**: StandardScaler for feature normalization

### Model Implementation
- **Traditional ML**: Logistic Regression, Random Forest, SVM with GridSearchCV
- **Deep Learning**: Neural networks using Sequential and Functional APIs
- **Hyperparameter tuning**: 5-fold cross-validation with systematic parameter search
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Key Findings
- **Academic performance** (final grades) is the strongest predictor
- **Study habits and family support** significantly impact predictions
- **Traditional ML models** slightly outperform deep learning approaches
- **Random Forest** achieved the best overall performance

## 📈 Generated Outputs

The notebook automatically generates:
- **Visualizations**: ROC curves, confusion matrices, feature importance plots
- **CSV files**: Experiment results and model comparisons
- **Analysis reports**: Error analysis and performance insights
- **Training histories**: Learning curves for deep learning models

## 🔧 Technical Details

### Dependencies
- **Python 3.7+**
- **NumPy, Pandas**: Data manipulation
- **Scikit-learn**: Traditional machine learning
- **TensorFlow 2.x**: Deep learning models
- **Matplotlib, Seaborn**: Visualizations
- **Plotly**: Interactive plots

### Reproducibility
- **Random seeds**: Set to 42 for consistent results
- **Dependencies**: All packages listed in notebook
- **Data source**: UCI ML Repository (automatic download)
- **Documentation**: Complete step-by-step explanations

## 📚 References

This project is based on the UCI Student Performance Dataset:
- **Source**: UCI ML Repository
- **Citation**: P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. FUBUTEC 2008.
- **URL**: https://archive.ics.uci.edu/ml/datasets/Student+Performance
