"""
Step 1: Load and Clean Real Credit Data
Days 1-2 of implementation

Loads train.csv and test.csv, performs data cleaning and basic EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create directories
for dir_name in ['data', 'visualizations', 'models', 'reports']:
    Path(dir_name).mkdir(exist_ok=True)

print("="*80)
print("STEP 1: LOADING REAL CREDIT DATA")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n📂 Loading datasets...")
train = pd.read_csv('train.csv', low_memory=False)
test = pd.read_csv('test.csv', low_memory=False)

print(f"✅ Train: {len(train):,} rows × {len(train.columns)} columns")
print(f"✅ Test:  {len(test):,} rows × {len(test.columns)} columns")

# ============================================================================
# 2. TARGET VARIABLE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TARGET DISTRIBUTION")
print("="*80)

target_counts = train['Credit_Score'].value_counts()
print("\nCredit Score Distribution:")
print(target_counts)
print(f"\nPercentages:")
for score, count in target_counts.items():
    pct = count / len(train) * 100
    print(f"  {score:10s}: {count:6,} ({pct:5.1f}%)")

# Visualize target distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='Credit_Score', order=['Poor', 'Standard', 'Good'])
plt.title('Credit Score Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Credit Score Category')
plt.ylabel('Count')
for i, v in enumerate(target_counts.sort_index()):
    plt.text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/01_target_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: visualizations/01_target_distribution.png")

# ============================================================================
# 3. DATA STRUCTURE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DATA STRUCTURE")
print("="*80)

# Check for time-series structure
unique_customers = train['Customer_ID'].nunique()
avg_months = len(train) / unique_customers

print(f"\n📊 Time-Series Structure:")
print(f"   - Unique customers: {unique_customers:,}")
print(f"   - Total records: {len(train):,}")
print(f"   - Avg records per customer: {avg_months:.1f}")
print(f"   → This is PANEL DATA (customer × time)")

months_per_customer = train.groupby('Customer_ID').size()
print(f"\n   - Min records per customer: {months_per_customer.min()}")
print(f"   - Max records per customer: {months_per_customer.max()}")
print(f"   - Median records per customer: {months_per_customer.median():.0f}")

# ============================================================================
# 4. DATA QUALITY CHECK
# ============================================================================

print("\n" + "="*80)
print("DATA QUALITY ANALYSIS")
print("="*80)

# Function to clean numeric columns
def detect_dirty_values(series):
    """Detect non-numeric values in supposedly numeric columns"""
    if series.dtype == 'object':
        # Check for special characters
        dirty_mask = series.astype(str).str.contains('_|#|@|!|\*|\$', na=False, regex=True)
        return dirty_mask.sum()
    return 0

print("\n⚠️  Checking for data quality issues...\n")

issues_found = []

# Check key numeric columns
numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Outstanding_Debt',
                'Credit_Utilization_Ratio', 'Amount_invested_monthly', 'Monthly_Balance']

for col in numeric_cols:
    if col in train.columns:
        dirty_count = detect_dirty_values(train[col])
        if dirty_count > 0:
            issues_found.append((col, dirty_count))
            print(f"   ⚠️  {col}: {dirty_count:,} dirty values")

# Check for missing values
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"\n   ⚠️  Missing values found in {len(missing)} columns:")
    for col, count in missing.head(10).items():
        pct = count / len(train) * 100
        print(f"      - {col}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 5. CLEAN DATA
# ============================================================================

print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

def clean_numeric_column(series):
    """Clean a numeric column by removing special characters and converting to float"""
    if series.dtype == 'object':
        # Remove special characters and underscores
        cleaned = series.astype(str).str.replace('[^0-9.-]', '', regex=True)
        # Convert to numeric, coerce errors to NaN
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        return cleaned
    return series

print("\n🧹 Cleaning numeric columns...")

# Clean each numeric column
for col in numeric_cols:
    if col in train.columns:
        original_nulls = train[col].isnull().sum()
        train[col] = clean_numeric_column(train[col])
        new_nulls = train[col].isnull().sum()
        
        if new_nulls > original_nulls:
            print(f"   - {col}: Cleaned, {new_nulls - original_nulls} additional NaNs")
        
        # Fill NaNs with median for numeric columns
        if train[col].isnull().sum() > 0:
            median_val = train[col].median()
            train[col].fillna(median_val, inplace=True)

# Clean test set the same way
for col in numeric_cols:
    if col in test.columns:
        test[col] = clean_numeric_column(test[col])
        if test[col].isnull().sum() > 0:
            median_val = test[col].median()
            test[col].fillna(median_val, inplace=True)

print("\n✅ Data cleaning complete!")

# ============================================================================
# 6. FEATURE OVERVIEW
# ============================================================================

print("\n" + "="*80)
print("FEATURE CATEGORIES")
print("="*80)

print("""
📊 Available Features (28 total):

1. DEMOGRAPHICS (4)
   - Age, Name, SSN, Occupation

2. FINANCIAL BASICS (3)
   - Annual_Income, Monthly_Inhand_Salary, Amount_invested_monthly

3. CREDIT ACCOUNTS (2)
   - Num_Bank_Accounts, Num_Credit_Card

4. LOANS (4)
   - Interest_Rate, Num_of_Loan, Type_of_Loan, Total_EMI_per_month

5. PAYMENT BEHAVIOR (4)
   - Delay_from_due_date, Num_of_Delayed_Payment, Payment_of_Min_Amount, Payment_Behaviour

6. CREDIT HEALTH (5)
   - Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix
   - Credit_Utilization_Ratio, Credit_History_Age

7. FINANCIAL STATUS (2)
   - Outstanding_Debt, Monthly_Balance

8. TIME (1)
   - Month

9. TARGET (1)
   - Credit_Score (Good/Standard/Poor)
""")

# ============================================================================
# 7. BASIC STATISTICS
# ============================================================================

print("="*80)
print("KEY STATISTICS")
print("="*80)

# Select numeric columns for stats
numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [c for c in numeric_features if c not in ['ID']]

print(f"\nNumeric features: {len(numeric_features)}")
print("\nSample statistics:")
print(train[numeric_features[:5]].describe())

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("QUICK CORRELATION CHECK")
print("="*80)

# Encode target for correlation
target_encoded = train['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

# Calculate correlations with target
correlations = pd.DataFrame()
for col in numeric_features[:15]:  # Top 15 features
    if col in train.columns:
        corr = train[col].corr(target_encoded)
        correlations = pd.concat([correlations, pd.DataFrame({
            'Feature': [col],
            'Correlation': [corr]
        })], ignore_index=True)

correlations = correlations.sort_values('Correlation', key=abs, ascending=False)
print("\nTop features correlated with Credit Score:")
print(correlations.head(10).to_string(index=False))

# ============================================================================
# 9. SAVE CLEANED DATA
# ============================================================================

print("\n" + "="*80)
print("SAVING CLEANED DATA")
print("="*80)

train.to_csv('data/train_cleaned.csv', index=False)
test.to_csv('data/test_cleaned.csv', index=False)

print(f"\n✅ Saved cleaned datasets:")
print(f"   - data/train_cleaned.csv ({len(train):,} rows)")
print(f"   - data/test_cleaned.csv ({len(test):,} rows)")

# ============================================================================
# 10. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
✅ Data Loading Complete!

Dataset Overview:
  - Training samples: {len(train):,}
  - Test samples: {len(test):,}
  - Unique customers: {unique_customers:,}
  - Features: {len(train.columns)}
  - Target classes: {len(target_counts)}

Target Distribution:
  - Poor: {target_counts.get('Poor', 0):,} ({target_counts.get('Poor', 0)/len(train)*100:.1f}%)
  - Standard: {target_counts.get('Standard', 0):,} ({target_counts.get('Standard', 0)/len(train)*100:.1f}%)
  - Good: {target_counts.get('Good', 0):,} ({target_counts.get('Good', 0)/len(train)*100:.1f}%)

Next Steps:
  1. Run: python 02_feature_engineering.py
  2. Create advanced features from this data
  3. Aggregate time-series per customer
""")

print("="*80)
print("STEP 1 COMPLETE! ✅")
print("="*80)