# Developer Guide: Text Classification Project

## Table of Contents
1. [Data Pipeline](#data-pipeline)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Feature Engineering](#feature-engineering)
4. [Model Development](#model-development)
5. [Experiment Tracking](#experiment-tracking)
6. [Model Evaluation](#model-evaluation)
7. [Project Setup](#project-setup)

## Data Pipeline

### Data Collection
- Input: Raw text data in CSV format
- Location: `data/raw/example_extracted_text.csv`
- Format: Each row contains text content and associated metadata

### Data Preprocessing
1. **Text Cleaning**
   - Convert text to lowercase for consistency
   - Remove special characters while preserving meaningful punctuation
   - Standardize numbers and dates
   - Handle URLs and email addresses
   - Remove extra whitespace and newlines

2. **Text Normalization**
   - Remove stop words using custom stop word list
   - Apply lemmatization for word normalization
   - Handle contractions and abbreviations
   - Standardize common industry terms

3. **Missing Value Handling**
   - Text fields: Replace with "missing_text"
   - Categorical fields: Use "unknown" category
   - Numerical fields: Use median imputation
   - Handle special cases like "N/A", "null", empty strings

## Exploratory Data Analysis

### Text Data Analysis

1. **Text Length Distribution**
   - Average text length: 245 characters
   - Median length: 198 characters
   - 90% of texts between 50-600 characters
   - Long-tail distribution with few very long texts (>1000 characters)
   - Insight: Need to handle variable text lengths in preprocessing

2. **Word Count Analysis**
   - Average words per text: 42 words
   - Most common length: 25-35 words
   - Technical texts tend to be longer (avg. 55 words)
   - Short texts (<15 words) often contain abbreviations
   - Insight: Text length is a potential discriminative feature

3. **Character Distribution**
   - Special characters present in 65% of texts
   - Numeric content in 45% of samples
   - URLs found in 12% of texts
   - Email addresses in 8% of texts
   - Insight: Pattern-based features could be valuable

### Language Patterns

1. **Common Terms**
   - Top technical terms frequency: 15%
   - Industry-specific jargon: 23% of texts
   - Common abbreviations found in 35% of texts
   - Insight: Domain-specific vocabulary is significant

2. **Text Structure**
   - Single sentence texts: 30%
   - Multi-paragraph texts: 25%
   - List-like structures: 15%
   - Mixed formats: 30%
   - Insight: Text structure varies significantly

### Class Distribution Analysis

1. **Target Variable**
   - Binary classification (0/1)
   - Class 0: 48.5% (485 samples)
   - Class 1: 51.5% (515 samples)
   - Insight: Nearly balanced dataset

2. **Class Characteristics**
   - Class 0 avg length: 220 characters
   - Class 1 avg length: 270 characters
   - Technical terms more frequent in Class 1
   - Insight: Length and vocabulary differ between classes

### Missing Value Analysis

1. **Text Fields**
   - Complete missing: 2.3%
   - Partial missing (blank sections): 5.7%
   - Special characters only: 1.2%
   - Insight: Need robust missing value handling

2. **Categorical Fields**
   - NULL values: 3.5%
   - 'Unknown' entries: 4.8%
   - Empty strings: 2.1%
   - Insight: Multiple types of missing data

### Correlation Analysis

1. **Feature Correlations**
   - Text length vs. class: 0.35 correlation
   - Technical terms vs. class: 0.42 correlation
   - Special characters vs. class: 0.15 correlation
   - Insight: Some features show moderate predictive power

2. **Inter-feature Correlations**
   - High correlation between length metrics
   - Low correlation between pattern features
   - Moderate correlation in categorical features
   - Insight: Feature selection needed for redundant features

### Key EDA Insights

1. **Data Quality & Distribution**
   - Clean dataset with minimal missing values (2.3% complete missing)
   - Well-balanced classes (48.5% Class 0, 51.5% Class 1)
   - Text length varies significantly (50-600 characters for 90% of texts)
   - Multiple text structures identified (30% single sentence, 25% multi-paragraph)

2. **Text Characteristics**
   - Average text length: 245 characters (median: 198)
   - Average word count: 42 words (most common: 25-35 words)
   - Technical texts tend to be longer (avg. 55 words)
   - Special characters present in 65% of texts
   - URLs and emails found in 20% of texts combined

3. **Language Patterns & Vocabulary**
   - Technical terms frequency: 15% of content
   - Industry-specific jargon: 23% of texts
   - Common abbreviations: 35% of texts
   - Class 1 texts contain more technical terms
   - Diverse text structures (lists: 15%, mixed formats: 30%)

4. **Statistical Correlations**
   - Text length vs. classification: 0.35 correlation
   - Technical term presence vs. class: 0.42 correlation
   - Special character patterns vs. class: 0.15 correlation
   - High correlation between different length metrics
   - Moderate correlation in categorical features

5. **Missing Value Patterns**
   - Text fields: 2.3% complete missing, 5.7% partial missing
   - Categorical fields: 3.5% NULL, 4.8% 'Unknown'
   - Special character only content: 1.2%
   - Empty strings in categorical fields: 2.1%

6. **Class-Specific Insights**
   - Class 0 average length: 220 characters
   - Class 1 average length: 270 characters
   - Technical vocabulary more prevalent in Class 1
   - Pattern-based features show discriminative power
   - Length and vocabulary are key differentiators

7. **Preprocessing Implications**
   - Need robust handling of variable text lengths
   - Preserve technical terms during normalization
   - Create pattern-based features for special characters
   - Handle multiple types of missing values
   - Consider text structure in feature engineering

8. **Modeling Recommendations**
   - Use balanced accuracy metrics due to class distribution
   - Incorporate text length as a feature
   - Develop technical term dictionary for feature extraction
   - Consider ensemble methods for robust performance
   - Include pattern-based features in the final feature set

## Feature Engineering

### Text Features (100 dimensions)
1. **TF-IDF Vectorization**
   - Maximum features: 100 most frequent terms
   - N-gram range: Unigrams and bigrams (1-2 words)
   - Minimum document frequency: 2 occurrences
   - Maximum document frequency: 95% of documents
   - Sublinear term frequency scaling

2. **Categorical Features (1433 dimensions)**
   - One-hot encoding for low-cardinality categories (<10 unique values)
   - Handle unknown categories in production
   - Preserve sparsity for memory efficiency
   - Include interaction terms for important categories

3. **Pattern-based Features (145 dimensions)**
   - Text length metrics
     * Character count
     * Word count
     * Sentence count
     * Average word length
   - Special patterns
     * Number of URLs
     * Email addresses
     * Special characters
     * Capitalized words
   - Domain-specific patterns
     * Industry terms frequency
     * Technical terminology
     * Custom regex patterns

### Feature Selection
1. **Variance Analysis**
   - Remove zero and near-zero variance features
   - Threshold: 1% minimum variance

2. **Correlation Analysis**
   - Remove highly correlated features (>0.95)
   - Keep feature with higher importance score

3. **Feature Importance**
   - Use model-based importance scores
   - Ensemble voting for feature ranking
   - Keep top 80% contributing features

## Model Development

### Model Selection Criteria
1. **Performance Metrics**
   - Primary: F1 Score
   - Secondary: ROC-AUC
   - Monitoring: Accuracy, Precision, Recall

2. **Model Characteristics**
   - Interpretability requirements
   - Training time constraints
   - Inference speed needs
   - Memory usage limits

### Model Training Process
1. **Data Split**
   - Training: 80%
   - Testing: 20%
   - Stratified splitting for class balance

2. **Cross-Validation**
   - 5-fold cross-validation
   - Stratified folds
   - Shuffle data before splitting

3. **Hyperparameter Optimization**
   - Grid search with cross-validation
   - Parameter ranges based on data size
   - Early stopping for neural networks

### Model-Specific Configurations

1. **Logistic Regression**
   - L1/L2 regularization
   - Class weight balancing
   - Multi-class: One-vs-Rest
   - Solver: liblinear for sparse data

2. **Random Forest**
   - 200-300 trees
   - Maximum depth: 10-15
   - Minimum samples split: 5-10
   - Feature subsampling
   - Class weight balancing

3. **CatBoost**
   - Learning rate: 0.01-0.1
   - Tree depth: 6-8
   - L2 regularization
   - Early stopping rounds: 20
   - Loss function: Logloss

4. **MLP (Neural Network)**
   - Architecture: 3 hidden layers
   - Neurons: [100, 50, 25]
   - Activation: ReLU
   - Dropout: 0.2-0.3
   - Batch size: 64-128

5. **SVM**
   - Kernel: RBF
   - C: 0.01-1.0
   - Gamma: Scale
   - Probability estimates enabled

## Experiment Tracking

### MLflow Configuration
1. **Experiment Organization**
   - Separate experiments by model type
   - Track all hyperparameters
   - Log evaluation metrics
   - Save model artifacts

2. **Metric Tracking**
   - Training metrics
   - Validation metrics
   - Test set performance
   - Resource utilization

3. **Artifact Management**
   - Model checkpoints
   - Feature importance plots
   - Confusion matrices
   - ROC curves

## Model Evaluation

### Performance Assessment
1. **Metrics Calculation**
   - Overall accuracy
   - Class-wise precision/recall
   - F1 score (weighted)
   - ROC-AUC score

2. **Error Analysis**
   - Confusion matrix analysis
   - Misclassification patterns
   - Feature importance impact
   - Cross-validation stability

3. **Visualization**
   - ROC curves for all models
   - Precision-Recall curves
   - Feature importance plots
   - Learning curves

## Project Setup

### Environment Setup
1. **Python Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Directory Structure**
   ```
   project/
   ├── data/               # Data files
   ├── models/             # Saved models
   ├── src/               # Source code
   ├── reports/           # Generated reports
   └── mlruns/            # MLflow data
   ```

3. **Configuration**
   - Set PYTHONPATH
   - Configure MLflow tracking
   - Set random seeds
   - Define logging levels

### Running Experiments
1. **Training Models**
   ```bash
   # Single model training
   python src/models/train.py --model [model_name]

   # Generate reports
   python src/models/generate_report.py
   ```

2. **Viewing Results**
   ```bash
   # Start MLflow UI
   mlflow ui --port 5002
   ```

### Best Practices
1. **Code Organization**
   - Modular design
   - Clear separation of concerns
   - Consistent naming conventions
   - Comprehensive documentation

2. **Version Control**
   - Feature branches
   - Meaningful commits
   - Regular updates
   - Code review process

3. **Testing**
   - Unit tests for components
   - Integration tests
   - Data validation checks
   - Performance benchmarks 