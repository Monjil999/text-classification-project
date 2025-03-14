# Text Classification Project 🚀

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A robust text classification system with multiple models, comprehensive evaluation metrics, and MLflow tracking.

<p align="center">
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="Python"/>
  </a>
  <a href="https://scikit-learn.org/" target="_blank">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  </a>
  <a href="https://www.mlflow.org/" target="_blank">
    <img src="https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/>
  </a>
  <a href="https://numpy.org/" target="_blank">
    <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy"/>
  </a>
  <a href="https://pandas.pydata.org/" target="_blank">
    <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  </a>
</p>

[Key Features](#key-features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Models](#models) •
[Results](#results)

</div>

## 📋 Project Workflow

<p align="center">
  <img src="assets/images/Text Classification Project_ End-to-End Summary - visual selection.png" alt="Project Workflow" width="80%">
</p>

## 🌟 Key Features

- **Multi-Model Support**: Logistic Regression, Random Forest, SVM, MLP, CatBoost
- **Advanced Feature Engineering**: TF-IDF, categorical encoding, pattern-based features
- **MLflow Integration**: Experiment tracking and model versioning
- **Comprehensive Evaluation**: ROC curves, confusion matrices, feature importance

## 📊 Model Performance

Best Model (Logistic Regression):
- Test Accuracy: 0.935
- F1 Score: 0.935
- ROC-AUC: 0.984

## 📈 EDA Insights

Our exploratory data analysis revealed several key insights that guided our modeling approach:

1. **Balanced Dataset & Text Distribution**
   - Nearly balanced classes (48.5% Class 0, 51.5% Class 1)
   - Text length varies from 50-600 characters (90% of data)
   - Average text length: 245 characters, with technical texts being longer (avg. 55 words)

2. **Strong Predictive Patterns**
   - Technical terms highly correlated with classification (0.42 correlation)
   - Text length moderately predictive (0.35 correlation)
   - Class 1 texts consistently longer (270 vs 220 characters) and more technical

3. **Text Characteristics**
   - Special characters present in 65% of texts
   - Technical jargon in 23% of content
   - Common abbreviations in 35% of texts
   - URLs and emails in 20% of texts combined

4. **Data Quality**
   - Minimal missing values (2.3% complete missing)
   - Multiple text structures (30% single sentence, 25% multi-paragraph)
   - Clean formatting with standardized patterns
   - Diverse vocabulary with domain-specific terms

For detailed EDA findings, see our [Developer Guide](DEVELOPER_GUIDE.md#exploratory-data-analysis).

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Monjil999/text-classification-project.git
cd text-classification-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models
python src/models/train.py --model logistic

# Start MLflow UI
mlflow ui --port 5002
```

## 📈 MLflow Integration

### Overview
MLflow is integrated into this project for comprehensive experiment tracking, model versioning, and performance monitoring. It provides a centralized platform to manage the entire machine learning lifecycle.

### Key Features
- Automated metric logging and parameter tracking
- Model artifact storage and versioning
- Interactive visualization dashboard
- Experiment comparison and analysis
- Run history and reproducibility

### Setup Instructions
```bash
# Start MLflow UI (default port)
mlflow ui --port 5002

# Alternative ports if 5002 is busy
mlflow ui --port 5003
mlflow ui --port 5004

# Access the dashboard
open http://localhost:5002
```

### Visualizations

#### 1. Model Performance Comparison
[![ROC Curves](assets/images/model_comparison_roc.png)](https://github.com/Monjil999/text-classification-project/blob/main/assets/images/model_comparison_roc.png)
*Comparative ROC curves showing Logistic Regression and CatBoost achieving superior AUC scores > 0.95*

#### 2. Confusion Matrix Analysis
[![Confusion Matrix](assets/images/confusion_matrix.png)](https://github.com/Monjil999/text-classification-project/blob/main/assets/images/confusion_matrix.png)
*Detailed confusion matrix highlighting high precision and recall across all classes*

#### 3. Feature Importance
[![Feature Analysis](assets/images/feature_importance.png)](https://github.com/Monjil999/text-classification-project/blob/main/assets/images/feature_importance.png)
*Top contributing features identified through model analysis, showing key text patterns and categorical variables*

#### 4. Experiment Dashboard
[![MLflow Dashboard](assets/images/mlflow_dashboard.png)](https://github.com/Monjil999/text-classification-project/blob/main/assets/images/mlflow_dashboard.png)
*Comprehensive experiment tracking interface displaying metrics, parameters, and artifacts*

### Future Enhancements
- Integration with model registry for production deployment
- Automated metric alerting and monitoring
- Custom metric visualization plugins
- Distributed training tracking
- API endpoint for programmatic access

## 🔧 Models

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|----------|
| Logistic Regression | 0.935 | 0.935 | 0.984 |
| CatBoost | 0.924 | 0.924 | 0.978 |
| Random Forest | 0.918 | 0.918 | 0.956 |
| MLP | 0.912 | 0.912 | 0.967 |
| SVM | 0.908 | 0.908 | 0.942 |

### Model Analysis Conclusion
Based on comprehensive evaluation across multiple metrics, Logistic Regression emerged as the best performing model with exceptional scores:
- Highest accuracy and F1 score (0.935)
- Superior ROC-AUC score (0.984)
- Excellent balance between precision and recall
- Computationally efficient with fast inference time

The CatBoost model follows closely behind, while other models show strong but slightly lower performance. This suggests that linear models work particularly well for our text classification task, possibly due to the effectiveness of our feature engineering pipeline.

## 📚 Documentation

For detailed documentation:
- [Developer Guide](DEVELOPER_GUIDE.md)

## 🔮 Future Scope & Vision

1. **Real-time Model Serving**
   - Deploy models using FastAPI/Flask for real-time predictions
   - Implement model versioning and A/B testing capabilities
   - Add request queuing and load balancing for high throughput

2. **Automated Model Retraining**
   - Set up data drift detection and monitoring
   - Implement automated retraining pipelines when performance degrades
   - Maintain versioned datasets for reproducibility

3. **Enhanced Feature Engineering**
   - Integrate transformer-based embeddings (BERT/RoBERTa)
   - Implement online feature computation for streaming data
   - Add domain-specific feature extractors for improved accuracy

4. **Production Monitoring Suite**
   - Track model health metrics and prediction quality
   - Set up alerting for performance degradation
   - Monitor resource utilization and response times

5. **Scalable Infrastructure**
   - Containerize the application using Docker
   - Set up Kubernetes for orchestration
   - Implement distributed training for larger datasets

6. **Security & Compliance**
   - Add model explainability reports for regulatory compliance
   - Implement data encryption and access controls
   - Set up audit logging for model predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Check the [Developer Guide](DEVELOPER_GUIDE.md) for setup instructions.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MLflow team for experiment tracking
- Scikit-learn community
- CatBoost developers 
