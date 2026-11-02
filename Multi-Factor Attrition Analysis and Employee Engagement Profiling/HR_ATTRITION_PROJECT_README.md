# Corporate HR Analytics & Workforce Attrition
## Multi-Factor Attrition Analysis and Employee Engagement Profiling

---

## 📋 Project Overview

### Business Problem
Organizations face significant challenges when experienced employees leave unexpectedly. **Attrition leads to:**
- Increased recruitment costs
- Loss of organizational knowledge
- Reduced productivity
- Project delivery delays

### Project Objective
Identify:
- ✅ Employees at risk of leaving
- ✅ Factors influencing attrition
- ✅ Workforce engagement patterns
- ✅ Retention improvement opportunities

---

## 📊 Dataset Information

### Primary Dataset
**IBM HR Analytics Employee Attrition & Performance Dataset**

### Key Features (18 Attributes)

| Feature | Description |
|---------|-------------|
| Age | Employee Age |
| Attrition | Yes/No (Target Variable) |
| BusinessTravel | Travel Frequency |
| Department | Employee Department (Sales, HR, R&D) |
| DistanceFromHome | Distance to Office |
| Education | Education Level |
| EnvironmentSatisfaction | Satisfaction Rating |
| Gender | Male/Female |
| JobInvolvement | Engagement Level |
| JobRole | Employee Role |
| JobSatisfaction | Satisfaction Score |
| MonthlyIncome | Salary |
| OverTime | Overtime Status (Yes/No) |
| TotalWorkingYears | Total Experience |
| WorkLifeBalance | Balance Rating |
| YearsAtCompany | Company Tenure |

---

## 🔍 Industry Problem Statement

Critical engineering and business teams often experience unexpected employee exits.

**Investigation Focus:**
- Why do employees leave?
- Which departments are most vulnerable?
- Does salary affect attrition rates?
- Does overtime contribute to resignations?

---

## 📦 Deliverables Checklist

### Technical Deliverables
- [ ] Google Colab Notebook (EDA + Analysis)
- [ ] GitHub Repository (Complete Code)
- [ ] IEEE Research Paper (10-11 pages)
- [ ] PowerPoint Presentation (20-25 slides)

### Business Deliverables
- [ ] Attrition Dashboard (Interactive Visuals)
- [ ] Employee Risk Profiling Report
- [ ] Retention Recommendations (Actionable Insights)

---

## 🛠️ Data Science Workflow

### Step 1: Load Dataset
```python
import pandas as pd
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()
df.shape
```

### Step 2: Data Understanding
```python
df.head()           # View first 5 rows
df.info()           # Data types and missing values
df.describe()       # Statistical summary
```

**Expected Findings:**
- Total number of employees
- Total features in dataset
- Attrition distribution (Yes/No counts)

### Step 3: Missing Value Analysis
```python
df.isnull().sum()
df.isnull().sum() / len(df) * 100  # Missing value percentage
```

**Question to Answer:** Are there any missing employee records?

### Step 4: Duplicate Detection
```python
df.duplicated().sum()
df[df.duplicated(keep=False)]  # Show duplicate rows
```

---

## 📈 Exploratory Data Analysis (EDA)

### Outlier Analysis (IQR Method)

**Apply on:**
- MonthlyIncome
- TotalWorkingYears
- YearsAtCompany

```python
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Visualization
import seaborn as sns
sns.boxplot(x=df['MonthlyIncome'])
plt.title('Monthly Income Outlier Detection')
plt.show()
```

**Expected Outputs:**
- Salary outliers identified
- Executive compensation patterns
- Outlier percentage

### Distribution Analysis

#### Employee Age Distribution
```python
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Employees')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

**Question:** Which age group dominates the workforce?

#### Monthly Income Distribution
```python
sns.histplot(df['MonthlyIncome'], bins=30, kde=True)
plt.title('Monthly Income Distribution')
plt.xlabel('Monthly Income ($)')
plt.ylabel('Frequency')
plt.show()
```

**Question:** Is salary distribution skewed?

#### Log Transformation (Address Skewness)
```python
import numpy as np
df['Income_Log'] = np.log1p(df['MonthlyIncome'])

# Compare before and after
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df['MonthlyIncome'], bins=30)
axes[0].set_title('Original Distribution')
axes[1].hist(df['Income_Log'], bins=30)
axes[1].set_title('Log-Transformed Distribution')
plt.show()
```

**Purpose:** Reduce skewness in salary distribution for better statistical analysis.

---

## 🔬 Univariate Analysis

### Department Distribution
```python
sns.countplot(x='Department', data=df, palette='Set2')
plt.title('Employee Count by Department')
plt.show()
```

### Attrition Distribution
```python
sns.countplot(x='Attrition', data=df, palette='coolwarm')
plt.title('Attrition Distribution')
attrition_counts = df['Attrition'].value_counts()
print(attrition_counts)
print(f"Attrition Rate: {(attrition_counts['Yes'] / len(df) * 100):.2f}%")
plt.show()
```

### Job Role Distribution
```python
sns.countplot(y='JobRole', data=df, palette='viridis')
plt.title('Employee Count by Job Role')
plt.show()
```

---

## 🔗 Bivariate Analysis

### Attrition vs Monthly Income
```python
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='Set2')
plt.title('Salary Comparison: Employees Who Left vs Stayed')
plt.ylabel('Monthly Income ($)')
plt.show()
```

**Question:** Do low-income employees leave more often?

### Attrition vs Overtime
```python
overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'])
print(overtime_attrition)

sns.heatmap(overtime_attrition, annot=True, fmt='d', cmap='RdYlGn_r')
plt.title('Attrition vs Overtime Status')
plt.show()
```

**Question:** Does overtime increase attrition?

### Attrition vs Work-Life Balance
```python
sns.boxplot(x='Attrition', y='WorkLifeBalance', data=df, palette='husl')
plt.title('Work-Life Balance: Employees Who Left vs Stayed')
plt.show()
```

**Question:** Does poor work-life balance increase resignations?

### Attrition vs Department
```python
dept_attrition = pd.crosstab(df['Department'], df['Attrition'])
dept_attrition_pct = dept_attrition.div(dept_attrition.sum(axis=1), axis=0) * 100

sns.heatmap(dept_attrition_pct, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Attrition Rate by Department (%)')
plt.show()
```

---

## 📊 Correlation Analysis

```python
# Select numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Calculate correlation
corr = numeric_df.corr()

# Visualize
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix: HR Analytics Features')
plt.tight_layout()
plt.show()
```

**Expected Output:** Relationships between:
- Salary and Attrition
- Experience and Attrition
- Satisfaction (Environment, Job) and Attrition
- Work-Life Balance and Attrition

---

## 📉 Statistical Validation

### Chi-Square Test (Categorical Independence)

**Question:** Is Overtime associated with Attrition?

```python
from scipy.stats import chi2_contingency

# Create contingency table
table = pd.crosstab(df['OverTime'], df['Attrition'])

# Perform chi-square test
chi2, p_value, dof, expected_freq = chi2_contingency(table)

print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of Freedom: {dof}")

if p_value < 0.05:
    print("✅ Significant relationship exists between Overtime and Attrition")
else:
    print("❌ No significant relationship found")
```

**Interpretation:**
- **p < 0.05:** Significant relationship exists ✅
- **p > 0.05:** No significant relationship ❌

### ANOVA Test (Group Means Comparison)

**Question:** Does salary differ significantly across departments?

```python
from scipy.stats import f_oneway

# Group salary by department
sales_salary = df[df['Department'] == 'Sales']['MonthlyIncome']
hr_salary = df[df['Department'] == 'Human Resources']['MonthlyIncome']
rd_salary = df[df['Department'] == 'Research & Development']['MonthlyIncome']

# Perform ANOVA
f_statistic, p_value = f_oneway(sales_salary, hr_salary, rd_salary)

print(f"F-Statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("✅ Significant salary differences exist across departments")
else:
    print("❌ No significant salary differences across departments")
```

### T-Test (Two Group Comparison)

**Example:** Do employees who left have different average salaries?

```python
from scipy.stats import ttest_ind

left = df[df['Attrition'] == 'Yes']['MonthlyIncome']
stayed = df[df['Attrition'] == 'No']['MonthlyIncome']

t_stat, p_value = ttest_ind(left, stayed)

print(f"T-Statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Mean Salary (Left): ${left.mean():.2f}")
print(f"Mean Salary (Stayed): ${stayed.mean():.2f}")
```

---

## 🎨 Required Visualizations

### Basic Visualizations
- ✅ Histogram (Age, Income distributions)
- ✅ Bar Chart (Department, Job Role counts)
- ✅ Pie Chart (Attrition rate percentage)

### Intermediate Visualizations
- ✅ Box Plot (Outlier detection, comparisons)
- ✅ Violin Plot (Distribution shapes)
- ✅ Heatmap (Correlation matrix, crosstabs)

### Advanced Visualizations
- ✅ Pairplot (Multi-variable relationships)
- ✅ Attrition Dashboard (KPIs + Metrics)
- ✅ Employee Segmentation Dashboard (Risk profiles)

---

## 🎯 Key Business Questions to Answer

| Q# | Question | Expected Finding |
|----|-----------|--------------------|
| Q1 | Which department experiences highest attrition? | Identify high-risk department |
| Q2 | Does overtime contribute to resignations? | Quantify overtime-attrition link |
| Q3 | Which employee group is most likely to leave? | Define high-risk segments |
| Q4 | Does salary significantly affect attrition? | Show salary threshold |
| Q5 | Does work-life balance influence retention? | Link satisfaction to attrition |

---

## 💡 Expected Insights

Students should identify:

1. **Insight 1: High-Risk Employee Groups**
   - Age range with highest attrition
   - Experience level most likely to leave
   - Job roles vulnerable to attrition

2. **Insight 2: Departments Requiring Retention Programs**
   - Department-wise attrition rates
   - Specific roles within departments
   - Root causes by department

3. **Insight 3: Salary Thresholds Linked to Attrition**
   - Critical salary points
   - Income distribution of those who left
   - Recommended salary ranges

4. **Insight 4: Impact of Overtime on Resignations**
   - Percentage of overtime workers who left
   - Statistical significance of overtime
   - Overtime burden by department

5. **Insight 5: Work-Life Balance Effects**
   - Balance ratings of those who left
   - Correlation with other satisfaction factors
   - Improvement recommendations

---

## 🚀 Model Building Extensions

### Easy Level: HR Analytics Dashboard
**Components:**
- Total Attrition Rate (%)
- Average Salary by Department
- Total Employee Count
- Department-wise Attrition Rates
- Age Group Distribution
- Job Role Breakdown

```python
# Sample KPI Calculation
total_employees = len(df)
attrition_count = len(df[df['Attrition'] == 'Yes'])
attrition_rate = (attrition_count / total_employees) * 100

print(f"📊 KEY PERFORMANCE INDICATORS")
print(f"Total Employees: {total_employees}")
print(f"Employees Left: {attrition_count}")
print(f"Attrition Rate: {attrition_rate:.2f}%")
print(f"Average Monthly Income: ${df['MonthlyIncome'].mean():.2f}")
```

### Medium Level: Employee Attrition Prediction Model

**Target Variable:** Attrition (Yes/No)

**Algorithms:**
- Logistic Regression
- Decision Tree
- Random Forest

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Data preprocessing
le = LabelEncoder()
df['Attrition_encoded'] = le.fit_transform(df['Attrition'])

# Feature selection and train-test split
X = df.drop(['Attrition', 'Attrition_encoded'], axis=1)
y = df['Attrition_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print(f"✅ Model Accuracy: {accuracy:.2%}")
print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred)}")
```

**Output:**
- Will Leave (Prediction: 1)
- Will Stay (Prediction: 0)

### Advanced Level: Employee Retention Recommendation Engine

**Inputs:**
- Salary Level
- Overtime Status
- Work-Life Balance Score
- Job Satisfaction Score

**Outputs:**
- Risk Level: Low / Medium / High
- Recommended Actions

```python
def retention_recommendation(salary, overtime, wlb, job_satisfaction):
    """
    Generate retention recommendations based on employee profile
    """
    risk_score = 0
    recommendations = []
    
    # Salary assessment
    if salary < df['MonthlyIncome'].quantile(0.25):
        risk_score += 3
        recommendations.append("💰 Salary Revision")
    
    # Overtime assessment
    if overtime == 'Yes':
        risk_score += 2
        recommendations.append("⏰ Flexible Work Arrangements")
    
    # Work-life balance assessment
    if wlb <= 2:
        risk_score += 2
        recommendations.append("🏠 Flexible Working Hours/WFH")
    
    # Job satisfaction assessment
    if job_satisfaction <= 2:
        risk_score += 2
        recommendations.append("📚 Career Development/Training")
    
    # Risk categorization
    if risk_score >= 6:
        risk_level = "🔴 HIGH RISK"
        recommendations.append("🎯 Immediate Intervention Required")
    elif risk_score >= 3:
        risk_level = "🟡 MEDIUM RISK"
    else:
        risk_level = "🟢 LOW RISK"
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'recommendations': recommendations
    }

# Example usage
result = retention_recommendation(
    salary=4000, 
    overtime='Yes', 
    wlb=2, 
    job_satisfaction=2
)

print(f"Risk Level: {result['risk_level']}")
print(f"Risk Score: {result['risk_score']}/10")
print(f"Recommended Actions:")
for action in result['recommendations']:
    print(f"  • {action}")
```

---

## 📁 GitHub Repository Structure

```
HR_Attrition_Analytics_Project/
│
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Raw dataset
│   └── data_dictionary.md             # Feature descriptions
│
├── notebooks/
│   ├── 01_Data_Loading_Understanding.ipynb
│   ├── 02_Data_Cleaning_EDA.ipynb
│   ├── 03_Univariate_Analysis.ipynb
│   ├── 04_Bivariate_Analysis.ipynb
│   ├── 05_Correlation_Statistical_Tests.ipynb
│   ├── 06_Model_Building.ipynb
│   └── 07_Insights_Recommendations.ipynb
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── eda_functions.py
│   ├── statistical_tests.py
│   ├── model_training.py
│   └── retention_engine.py
│
├── outputs/
│   ├── processed_data.csv
│   ├── feature_importance.csv
│   └── model_metrics.json
│
├── visualizations/
│   ├── distributions/
│   ├── correlations/
│   ├── comparisons/
│   └── dashboards/
│
├── presentation/
│   └── HR_Attrition_Analytics_Presentation.pptx
│
├── ieee_report/
│   └── IEEE_Research_Paper.docx
│
└── .gitignore
```

---

## 📄 IEEE Paper Structure

Follow this 11-section format:

```
1. ABSTRACT (150-250 words)
   - Problem statement
   - Methodology
   - Key findings
   - Business impact

2. INTRODUCTION (2-3 pages)
   - Industry context
   - Problem motivation
   - Project objectives
   - Contributions

3. LITERATURE REVIEW (2 pages)
   - Related attrition studies
   - Machine learning approaches
   - Dashboard design practices
   - Gaps in current research

4. DATASET DESCRIPTION (1 page)
   - Dataset source and size
   - Features and data types
   - Data distribution
   - Preprocessing steps

5. DATA CLEANING (1 page)
   - Missing value handling
   - Duplicate detection
   - Outlier treatment
   - Data quality improvements

6. EXPLORATORY DATA ANALYSIS (2-3 pages)
   - Univariate analysis
   - Bivariate analysis
   - Distribution patterns
   - Key observations

7. STATISTICAL VALIDATION (1-2 pages)
   - Chi-square test results
   - ANOVA findings
   - P-value interpretations
   - Significance testing

8. RESULTS (2-3 pages)
   - Model performance metrics
   - Feature importance rankings
   - Risk profiling outputs
   - Dashboard KPIs

9. RECOMMENDATIONS (1-2 pages)
   - Department-specific strategies
   - Salary recommendations
   - Work-life balance initiatives
   - Timeline and ROI

10. CONCLUSION (1 page)
    - Summary of findings
    - Business impact
    - Future work

11. REFERENCES (0.5-1 page)
    - Academic citations
    - Dataset reference
    - Tool documentation
```

---

## 🎤 Mock Interview Questions

### Technical Questions

1. **Why did you choose the IBM HR dataset?**
   - Relevance to business problems
   - Completeness and real-world applicability
   - Multiple feature types for analysis

2. **Why use IQR for salary analysis?**
   - Robust to outliers
   - Works with skewed distributions
   - Standard statistical approach
   - Better than Z-score for non-normal data

3. **Why apply log transformation?**
   - Reduce right skewness in income
   - Normalize distribution for modeling
   - Improve statistical test assumptions
   - Better visualization and interpretation

4. **Why choose Chi-Square for overtime analysis?**
   - Both variables are categorical
   - Tests independence of two categories
   - Appropriate null hypothesis testing
   - Provides p-value for significance

5. **Why use ANOVA for department-wise salary comparison?**
   - Compares means across 3+ groups
   - Tests if differences are significant
   - More efficient than multiple t-tests
   - Assumes normal distribution and equal variance

### Business Questions

1. **What is the biggest reason employees leave?**
   - Expected Answer: Based on analysis—e.g., "Low salary, excessive overtime, poor work-life balance in specific departments"
   - Support with correlation coefficients and statistical tests

2. **Which department needs intervention?**
   - Expected Answer: Identify department with highest attrition rate
   - Provide percentage and specific roles
   - Link to salary/overtime/satisfaction data

3. **What retention strategy would you recommend?**
   - Answer based on insights: salary revision, flexible work, training programs
   - Prioritize high-risk groups
   - Estimate cost-benefit

### Communication Questions

1. **Explain your project in 60 seconds.**
   - Problem: Employee attrition costs organizations significantly
   - Solution: Analyzed IBM HR dataset with Python/SQL
   - Methods: EDA, correlation analysis, statistical tests
   - Output: Risk profiles + dashboards + 3 key recommendations
   - Impact: Helps HR teams prioritize retention efforts

2. **Explain one important visualization.**
   - Choose your best chart (e.g., attrition vs salary boxplot)
   - Describe what it shows
   - Interpret the pattern
   - Link to business action

3. **Explain one actionable business recommendation.**
   - Start with insight (e.g., employees earning <X leave at Y% rate)
   - Propose action (e.g., implement salary review for this band)
   - Estimate impact (e.g., could reduce attrition by Z%)

---

## 🎯 Placement Roles Aligned

This project aligns with these job roles:

- 📊 **HR Analyst**
- 👥 **People Analytics Specialist**
- 💼 **Business Analyst**
- 📈 **Data Analyst**
- 🎯 **Workforce Planning Analyst**
- 📉 **BI Analyst**
- 🤖 **Data Scientist** (Entry-level)

---

## 💪 Why This Project Excels

✅ **Highly Relatable:** Everyone understands employee satisfaction, salary, overtime, and retention

✅ **Strong Visualizations:** Produces compelling dashboards with clear business insights

✅ **Interview-Friendly:** Easy to explain, with clear problem-solution narrative

✅ **Multiple Complexity Levels:** Can be simplified or extended based on time/requirements

✅ **Real Business Value:** Directly applicable to HR departments

✅ **Comprehensive Skill Demonstration:** EDA → Statistics → Modeling → Visualization → Recommendations

---

## 📝 Next Steps

1. ✅ Download IBM HR Analytics dataset
2. ✅ Set up Google Colab notebook
3. ✅ Follow data science workflow step-by-step
4. ✅ Create visualizations for each analysis
5. ✅ Perform statistical tests and document findings
6. ✅ Build prediction model (easy/medium/advanced level)
7. ✅ Generate insights and business recommendations
8. ✅ Create interactive dashboard
9. ✅ Write IEEE research paper
10. ✅ Prepare PowerPoint presentation
11. ✅ Push complete project to GitHub
12. ✅ Practice mock interview answers

---

**Project Created:** June 2026
**Status:** Ready for Implementation
**Difficulty Level:** Beginner-Friendly with Advanced Extensions

