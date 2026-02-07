# Data Directory

## Files

### Preprocessed Data
After running the notebook, this directory will contain:
- `train_processed.csv` - Preprocessed training data (features + target)
- `test_processed.csv` - Preprocessed test data (features only)

### Data Source
Raw data is automatically downloaded from:
- Training: https://luminwin.github.io/ASASF/train.rds
- Test: https://luminwin.github.io/ASASF/test.rds

## Feature Categories

### Demographics
- RIAGENDR: Gender (1=Male, 2=Female)
- RIDAGEYR: Age in years
- RIDRETH: Race/Ethnicity

### Body Measurements (BMX prefix)
- BMXBMI: Body Mass Index (kg/m²)
- BMXWAIST: Waist Circumference (cm)
- BMXWT: Weight (kg)
- BMXHT: Standing Height (cm)

### Dietary/Nutrition (DR prefix)
- Various nutrient intake variables
- Daily caloric intake
- Macronutrient consumption

### Health Indicators
- Blood pressure measurements
- Laboratory results
- Smoking status (SMQ prefix)

### Target Variable
- **LBDHDD_outcome**: Direct HDL-Cholesterol (mg/dL)
  - Noise-perturbed for privacy
  - Higher values indicate better cardiovascular health
  - Normal range: 40-60 mg/dL (men), 50-60 mg/dL (women)

## Data Preprocessing Steps

1. **Missing Value Handling**:
   - Numeric: Median imputation
   - Categorical: Most frequent value imputation

2. **Feature Scaling**:
   - StandardScaler for numeric features
   - Mean=0, Std=1

3. **Encoding**:
   - One-hot encoding for categorical variables
   - Handle unknown categories

## Notes
- Total features: 100+ variables
- Training samples: ~7,000+
- Test samples: ~3,000+
- Missing data varies by feature (0-90% range)
