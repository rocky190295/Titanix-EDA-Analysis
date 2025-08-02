# Titanic Exploratory Data Analysis (EDA)
This project contains a comprehensive exploratory data analysis of the famous Titanic dataset. The primary goal is to understand the factors that contributed to the survival or demise of passengers on the ill-fated voyage.

## Introduction
The Titanic dataset is a classic for beginners in data science and machine learning. This notebook performs a detailed exploratory data analysis to uncover patterns and relationships within the data, which can later be used for predictive modeling. The original dataset used in this analysis is [test.csv](https://github.com/rocky190295/Titanix-EDA-Analysis/blob/main/train.csv)

## Data Dictionary
The dataset contains the following variables:
| Variable | Definition | Key    |
|----------|----------|----------|
| `survival` | Survival | `0` = No `1`= Yes |
| `pclass` | Ticket class | `1` = 1st, `2`=2nd, `3`=3rd Class |
| `sex`    | Gender        |               |
| `Age`    | Age in years  |               |
| `sibsp`  | # of siblings/spouses aboard the Titanic |  |
| `parch`  | # of parents/children aboard the titanic |  |
| `ticket` | Ticket number |  |
| `fare`   | Ticket fare   |  |
| `cabin`  | Cabin Number  |  |
| `embarked` | Port of embarkation | `C`= Cherbourg, `Q`=Queenstown, `S` = Southampton |


## Key findings & Visualizations
## Missing Data Analysis
A crucial first step was to identify and handle missing values.
- Cabin: This feature has a significant number of missing values (over 77%), making it difficult to use directly.
- Age: Approximately 20% of the Age data is missing. This was handled by imputing the missing values.
- Embarked: Only 2 values were missing, which were filled with the mode.
  
The missing data was summarized and handled to ensure the dataset is clean for analysis.


## Target Variable Analysis
The `Survived` column is our target variable. The analysis revealed that only a small portion of passengers survived.
- Approximately 62% of the passengers did not survive (Survived = 0).
- Approximately 38% of the passengers survived (Survived = 1).
  
This is a classic example of an imbalanced dataset, which is an important consideration for later modeling.


## Survival by Sex and Pclass
The most striking finding is the strong correlation between `Sex`, `Pclass`, and `Survival`.
- **Gender Bias**: Females had a significantly higher survival rate than males. This aligns with the "women and children first" protocol.
- **Social Class**: Passengers in first class (`Pclass = 1`) had the highest survival rate, while those in third class (`Pclass = 3`) had the lowest.

This suggests that access to lifeboats and higher deck locations played a critical role.

The visualizations below confirm these patterns:
**Survival Count Plot:**
<img width="577" height="456" alt="1  Titanic Survival Count" src="https://github.com/user-attachments/assets/a85752a4-6ff3-4b00-9310-963c86efd1d2" />

Above bar graph is the visual of the survival rate which clearly distinguish between the rate of survivals

## Survival Rate by Gender
<img width="570" height="459" alt="2  Survival Rate By Gender" src="https://github.com/user-attachments/assets/73b8e1ef-83cc-48a3-be8c-1a2ab5d78489" />

The above Survival Rate By Gender plot also clearly shows that Female had significantly more survival rate than men. Meaning WOMEN are prioritized during rescue effort.

## Age Distribution and Survival
Age played a complex role. The age distribution for survivors and non-survivors shows that certain age groups had a better chance of survival.
- Young children and the elderly appeared to have a better chance of survival, particularly in higher classes.
- The majority of casualties were young to middle-aged adults.
- Age Distribution Plot:
<img width="865" height="472" alt="5  Age Distribution by Sruvival Status" src="https://github.com/user-attachments/assets/ec10626a-13db-448a-a07e-438f4bea7c26" />

## Survival by Family Size
Passengers traveling with a small to medium-sized family (`FamilySize` between 1 and 4) had a higher survival rate compared to those traveling alone or with large families.
- Travelers with large families faced a greater challenge in getting everyone to safety.
- Family Size Survival Plot:
<img width="571" height="456" alt="14  Survival rate by family size" src="https://github.com/user-attachments/assets/295c6e9f-7fcb-4888-ba4e-16616905182c" />

## Survival by Fare
The fare paid by the passengers shows a clear correlation with survival.
- Passengers who paid a higher fare, likely traveling in first or second class, had a much higher survival rate.
- This reinforces the finding that economic status and cabin class were major factors in who survived.
- Fare Distribution Plot:
<img width="700" height="470" alt="9  Fare Distribution" src="https://github.com/user-attachments/assets/acb49883-1b20-4e2c-86b6-a2a51ba09e03" />

---

## Feature Engineering
Several new features were created to improve the predictive power of the model. These steps involved transforming existing data to better represent the underlying relationships.
- **Title**: A new `Title` feature was extracted from the Name column. This provided valuable categorical information (e.g., 'Mr', 'Miss', 'Master', 'Mrs') that strongly correlated with `Sex` and `Age`. This feature was also used to impute missing Age values.
- **FamilySize**: The `SibSp` and `Parch` columns were combined to create a `FamilySize` feature, which represents the total number of family members a passenger had on board.
- **IsAlone**: A boolean IsAlone feature was created based on `FamilySize`. This feature distinguishes between passengers traveling by themselves and those traveling with a family, which was found to be a significant predictor of survival.

---

##  EDA Summary

Through this Exploratory Data Analysis, I gained several key insights:

- **Age and Survival**: Children and teenagers had a significantly higher chance of survival, aligning with evacuation priority norms.
- **Family Dynamics**: Passengers traveling with small families (2–4 members) had better survival chances compared to solo travelers or large families.
- **Social Status**: Titles extracted from names revealed that women and socially important passengers had much higher survival rates than men, particularly those with the title "Mr".
- **Being Alone Matters**: "IsAlone" was a strong indicator—solo travelers had a significantly lower survival rate.
- **Data Gaps Identified**: Some passengers had missing values for Age, Cabin, or Embarked, which were addressed through imputation or feature engineering.

These insights help us engineer meaningful features and prepare the data for predictive modeling.

---

## Dependencies
This project requires the following python libraries
- `pandas`
- `matplotlib`
- `seaborn`
