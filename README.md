# HDL Cholesterol Prediction using Machine Learning

**GitHub Repository**: https://github.com/rvargasm7/hdl-prediction-project

## Team
Rolando Vargas, Eleniz Espina, Bryce Leister

University of Miami
MAS 635 - Machine Learning Methods

## Project Overview
This project predicts HDL (High-Density Lipoprotein) cholesterol levels using demographic, dietary, behavioral, and body measurement variables from the NHANES (National Health and Nutrition Examination Survey) dataset.

**Course**: MAS 635 - Machine Learning Methods
**Institution**: University of Miami
**Semester**: Spring 2026

## Target Variable
- **LBDHDD_outcome**: Direct HDL-Cholesterol (mg/dL) - noise-perturbed for privacy

## Dataset
- **Source**: [ASA South Florida Student Data Challenge](https://luminwin.github.io/ASASF/)
- **Training samples**: 1,000 observations
- **Test samples**: 200 observations
- **Features**: 95 variables including demographics, body measurements, dietary intake, and health indicators

## Data Access

### Preprocessed Data
The preprocessed training and test datasets are available in the `data/` directory of this repository:

**GitHub Data Link**: https://github.com/rvargasm7/hdl-prediction-project/tree/main/data

Files:
- `train_processed.csv` - Preprocessed training data
- `test_processed.csv` - Preprocessed test data
- `README.md` - Data dictionary and description of features

### Raw Data
Original data can be downloaded from:
- Training: https://luminwin.github.io/ASASF/train.rds
- Test: https://luminwin.github.io/ASASF/test.rds

## Project Structure
```
MidtermProject/
├── README.md                              # Project documentation
├── HDL_Cholesterol_Prediction.ipynb      # Main analysis notebook
├── requirements.txt                       # Python dependencies
├── data/                                  # Data directory
│   ├── train_processed.csv
│   ├── test_processed.csv
│   └── data_dictionary.md
├── outputs/                               # Generated outputs
│   ├── pred.csv                          # Final predictions
│   └── figures/                          # Visualization outputs
├── report/                                # Project report
│   └── HDL_Prediction_Report.pdf
└── presentation/                          # Presentation materials
    └── HDL_Prediction_Slides.pdf
```

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Missing value analysis and visualization
- Target distribution analysis
- Correlation analysis
- Relationship exploration (HDL vs BMI, demographics, nutrients)

### 2. Data Preprocessing
- Missing value imputation (median for numeric, mode for categorical)
- Standardization of numeric features
- One-hot encoding of categorical variables
- Train-validation split (80-20)

### 3. Models Implemented

#### Baseline Models:
- Linear Regression
- Ridge Regression
- Elastic Net
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor

#### Deep Learning Models:
- Standard Tabular Neural Network (4 hidden layers with BatchNorm and Dropout)
- Advanced Neural Network with Skip Connections (Residual architecture)

### 4. Model Evaluation
- Metrics: RMSE, MAE, R²
- 5-fold Cross-Validation
- Weighted ensemble approach for final predictions

## Installation & Setup

### Requirements
- Python 3.8+
- See `requirements.txt` for full list of dependencies

### Installation
```bash
# Clone the repository
git clone https://github.com/rvargasm7/hdl-prediction-project.git
cd hdl-prediction-project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Notebook

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Open `HDL_Cholesterol_Prediction.ipynb`

3. Run all cells sequentially (Runtime → Run All)

**Note**: The notebook will:
- Download data automatically from the source
- Generate visualizations in the current directory
- Save final predictions to `pred.csv`

## Results

### Best Performing Models (Validation Set)
| Model | Validation RMSE | Validation MAE | R² |
|-------|----------------|----------------|-----|
| Gradient Boosting | 5.0312 | 3.9438 | 0.6963 |
| CatBoost | 5.0388 | 3.9370 | 0.6954 |
| XGBoost | 5.1887 | 4.1054 | 0.6770 |
| Elastic Net | 5.8996 | 4.6086 | 0.5825 |
| Ridge Regression | 5.9012 | 4.5957 | 0.5822 |
| Linear Regression | 5.9322 | 4.6188 | 0.5778 |
| Random Forest | 6.2469 | 4.8718 | 0.5319 |
| Basic Neural Network | 6.5471 | 5.1350 | 0.4858 |
| Advanced Neural Network | 7.3001 | 5.8338 | 0.3607 |

### Stacking Ensemble (Final Model)
| Component | OOF RMSE |
|-----------|----------|
| XGBoost (Optuna-tuned) | 4.7031 |
| CatBoost (Optuna-tuned) | 4.7318 |
| Gradient Boosting | 4.8308 |
| Random Forest | 5.8315 |
| **Stacked (Ridge meta-learner)** | **4.6434** |

### Cross-Validation RMSE (5-Fold)
| Model | Mean RMSE | Std |
|-------|-----------|-----|
| XGBoost (Optuna) | 4.6978 | - |
| CatBoost (Optuna) | 4.7277 | - |
| CatBoost (default) | 4.8103 | ±0.1370 |
| XGBoost (default) | 4.8477 | ±0.1587 |

### Key Findings
1. Strong negative correlation between HDL and body measurements (BMI, waist circumference)
2. Significant differences in HDL levels across demographic groups
3. Optuna hyperparameter tuning improved CatBoost CV from 4.81 to 4.73 and XGBoost from 4.85 to 4.70
4. Stacking ensemble with Ridge meta-learner achieved OOF RMSE of 4.6434, a 1.27% improvement over the best individual model

## Business Insights
- **Healthcare Applications**: Predictive model can help identify individuals at risk for cardiovascular disease
- **Risk Stratification**: Model enables early intervention for patients with predicted low HDL
- **Resource Allocation**: Healthcare providers can prioritize screening for high-risk populations
- **Lifestyle Recommendations**: Feature importance reveals modifiable risk factors

## Output Files
- `pred.csv`: Final predictions for test set (submission file)
- `*.png`: Visualization outputs from EDA and modeling

## Dependencies
Key packages used:
- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization
- scikit-learn: Machine learning models and preprocessing
- xgboost, catboost: Gradient boosting models
- tensorflow, keras: Deep learning framework
- pyreadr: Reading R data files

## References
1. [NHANES Program Overview](https://www.cdc.gov/nchs/nhanes/about/index.html)
2. [ASA South Florida Data Challenge](https://luminwin.github.io/ASASF/)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)
4. [CatBoost Documentation](https://catboost.ai/docs/)
5. [TensorFlow Documentation](https://www.tensorflow.org/)
