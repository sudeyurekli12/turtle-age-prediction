# Turtle Age Prediction — Non-Invasive Regression Modelling
 
**Course:** Introduction to Machine Learning — Tilburg University (2025)  
**Author:** Sude Yurekli (Individual Assignment)  
**Tools:** Python, scikit-learn, NumPy, pandas, seaborn, matplotlib
 
---
 
## The Problem
 
Determining a turtle's age typically requires slicing its shell to count growth rings — an invasive and time-consuming process. A marine wildlife conservation agency needs a faster, non-invasive method to estimate turtle age in order to regulate fishing and prevent overharvesting of a newly discovered species.
 
This project builds a regression model that predicts the number of shell growth rings (a proxy for age) from physical measurements like shell length, diameter, and weight.
 
---
 
## What I Did
 
### Data & Preprocessing
- Dataset: morphological measurements of turtles including shell length, diameter, weight, and gender
- Encoded the categorical `Sex` column using one-hot encoding
- Binned the target variable (`Rings`) using `np.digitize` for stratified splitting
- Split data into **80% training / 10% validation / 10% test** using `StratifiedShuffleSplit` to preserve ring distribution across splits
- Scaled numerical features with `StandardScaler`
 
### Models Compared
 
| Model | R² | MSE | MAE |
|---|---|---|---|
| Linear Regression | 0.474 | 5.52 | 1.66 |
| Polynomial Regression (degree 2) | **0.515** | **5.08** | **1.58** |
| KNN (k=10) | 0.508 | 5.16 | 1.57 |
 
Polynomial Regression (degree 2) performed best overall. Higher degrees (3–5) overfit badly on the validation set and were rejected.
 
### Hyperparameter Tuning
- **Polynomial Regression:** looped over degrees 2–5, selected best on validation MSE → degree 2
- **KNN:** looped over k = {1, 3, 6, 10}, selected best on validation MSE → k = 10
 
### Cross-Validation
Performed **10-fold cross-validation** on the best model (Polynomial degree 2):
- Collected R², MSE, and MAE per fold
- Identified best and worst performing folds
- Investigated over- and under-prediction patterns: the model struggled most with turtles at extreme age ranges, which is a common limitation with polynomial regression on skewed targets
 
### Feature Importance
In the best cross-validation fold, **shell weight** was identified as the most influential predictor of ring count (correlation: 0.63), consistent with the EDA findings.
 
---
 
## Key Findings
 
- All three models performed similarly, explaining roughly 47–52% of variance in ring count
- Linear relationships between shell measurements and age exist but are not strong enough for high-accuracy prediction
- Gender had minimal influence on ring count; immature turtles showed slightly lower median rings as expected
- A non-linear approach (e.g. Random Forest) would likely improve performance
 
---
 
## Honest Reflection
 
The dataset's features correlate moderately but not strongly with the target, suggesting unmeasured biological factors also drive turtle age. The polynomial model's degree 2 was the sweet spot — higher degrees overfit quickly given the dataset size. Future work could explore ensemble methods or additional features like habitat or diet data.
 
---
 
## Files
- `ML_1_ASS_Notebook.ipynb` — full notebook with EDA, modelling, cross-validation, error analysis, and results summary
- `final_model_performance.csv` — exported summary of all model metrics
