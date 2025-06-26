This project aims to build a machine learning model that can accurately predict whether a person is likely to develop heart disease based on a variety of medical attributes. Using a dataset containing clinical information 
like age, cholesterol level, chest pain type, and resting blood pressure, we preprocess the data, train multiple models, and evaluate their performance.
The final model — an **XGBoost Classifier** — is chosen for its superior accuracy and robustness. Additionally, the project includes an interactive component where users can input patient data manually (via terminal or Jupyter
widgets) and receive a prediction along with confidence probabilities for both outcomes: *heart disease* or *no heart disease*.
This project showcases the complete ML pipeline:
- Exploratory data analysis and visualization
- Feature engineering and one-hot encoding
- Model selection and hyperparameter tuning
- Evaluation using real test data
- Interactive, real-time prediction system
##Features Explained

The dataset includes the following attributes:
| Feature           | Description |
|------------------|-------------|
| `Age`            | Age of the patient (in years) |
| `Sex`            | Gender of the patient (`M` = Male, `F` = Female) |
| `ChestPainType`  | Type of chest pain: `TA` (Typical Angina), `ATA` (Atypical Angina), `NAP` (Non-Anginal Pain), `ASY` (Asymptomatic) |
| `RestingBP`      | Resting blood pressure (in mm Hg) |
| `Cholesterol`    | Serum cholesterol (in mg/dl) |
| `FastingBS`      | Fasting blood sugar (`1` if >120 mg/dl, else `0`) |
| `RestingECG`     | Resting electrocardiographic results: `Normal`, `ST` (having ST-T wave abnormality), `LVH` (showing probable/definite left ventricular hypertrophy) |
| `MaxHR`          | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (`Y` = Yes, `N` = No) |
| `Oldpeak`        | ST depression induced by exercise relative to rest |
| `ST_Slope`       | Slope of the peak exercise ST segment: `Up`, `Flat`, or `Down` |
| `HeartDisease`   | Target variable: `1` = heart disease, `0` = no heart disease |

##Techniques Used

##Data Preprocessing
- One-hot encoding for categorical variables (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`)
- Splitting data into training and validation sets using `train_test_split`
- Feature scaling not required due to tree-based models

##Exploratory Data Analysis (EDA)
- Distribution plots using `Seaborn` and `Matplotlib`
- Target balance inspection
- Correlation checks

##Model Training
- **DecisionTreeClassifier** – as a baseline
- **RandomForestClassifier** – for ensemble-based improvement
- **XGBoostClassifier** – chosen final model due to high accuracy and interpretability

##Model Evaluation
- Accuracy Score on both training and validation sets
- Probability-based predictions using `.predict_proba()`
- Classification metrics: precision, recall, F1-score (optional)

##Interactive Prediction
- Manual input of patient data via console or widgets
- Real-time prediction with class probabilities
- Feature alignment using one-hot encoding and `reindex()` to match model expectations
