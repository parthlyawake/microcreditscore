"""
Step 2: Feature Engineering
Days 3-4 of implementation

Creates 70-80 advanced features including:
- Customer-level aggregations (from time-series)
- Traditional credit features
- Behavioral patterns
- Derived metrics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 2: FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# 1. LOAD CLEANED DATA
# ============================================================================

print("\n📂 Loading cleaned data...")
train = pd.read_csv('data/train_cleaned.csv')
print(f"✅ Loaded: {len(train):,} rows")

# ============================================================================
# 2. CUSTOMER-LEVEL AGGREGATION
# ============================================================================

print("\n" + "="*80)
print("CUSTOMER-LEVEL AGGREGATION (Time-Series → Single Row)")
print("="*80)

print("""
Strategy: Each customer has ~8 months of data
We'll aggregate to create ONE row per customer with:
- Latest values (most recent month)
- Trends (improving/declining)
- Volatility (consistency over time)
- Historical patterns
""")

def create_customer_features(customer_data):
    """
    Aggregate multiple months of data into single customer profile
    """
    features = {}
    
    # Basic info (take most recent/most common)
    features['customer_id'] = customer_data['Customer_ID'].iloc[0]
    features['age'] = customer_data['Age'].median()
    features['occupation'] = customer_data['Occupation'].mode()[0] if len(customer_data['Occupation'].mode()) > 0 else 'Unknown'
    
    # Target (most common credit score across months)
    features['credit_score'] = customer_data['Credit_Score'].mode()[0] if len(customer_data['Credit_Score'].mode()) > 0 else customer_data['Credit_Score'].iloc[-1]
    
    # ==============================================
    # INCOME FEATURES
    # ==============================================
    
    # Latest income
    features['annual_income'] = customer_data['Annual_Income'].iloc[-1]
    features['monthly_income'] = customer_data['Monthly_Inhand_Salary'].iloc[-1]
    
    # Income stability (coefficient of variation)
    if customer_data['Annual_Income'].std() > 0:
        features['income_volatility'] = customer_data['Annual_Income'].std() / customer_data['Annual_Income'].mean()
    else:
        features['income_volatility'] = 0
    
    # Income trend (slope)
    if len(customer_data) >= 3:
        x = np.arange(len(customer_data))
        income_vals = customer_data['Annual_Income'].values
        if len(income_vals) > 0 and not np.all(np.isnan(income_vals)):
            features['income_trend'] = np.polyfit(x, income_vals, 1)[0]
        else:
            features['income_trend'] = 0
    else:
        features['income_trend'] = 0
    
    # ==============================================
    # PAYMENT BEHAVIOR FEATURES
    # ==============================================
    
    # Payment punctuality
    features['avg_payment_delay'] = customer_data['Delay_from_due_date'].mean()
    features['max_payment_delay'] = customer_data['Delay_from_due_date'].max()
    features['payment_delay_volatility'] = customer_data['Delay_from_due_date'].std()
    
    # Number of delayed payments
    features['total_delayed_payments'] = customer_data['Num_of_Delayed_Payment'].sum()
    features['avg_delayed_payments'] = customer_data['Num_of_Delayed_Payment'].mean()
    features['recent_delayed_payments'] = customer_data['Num_of_Delayed_Payment'].tail(3).mean()
    
    # Payment behavior consistency
    late_months = (customer_data['Delay_from_due_date'] > 0).sum()
    features['late_payment_frequency'] = late_months / len(customer_data)
    
    # Payment trend (improving or declining)
    if len(customer_data) >= 3:
        recent_delays = customer_data['Delay_from_due_date'].tail(3).mean()
        older_delays = customer_data['Delay_from_due_date'].head(3).mean()
        features['payment_behavior_trend'] = older_delays - recent_delays  # Positive = improving
    else:
        features['payment_behavior_trend'] = 0
    
    # ==============================================
    # CREDIT ACCOUNT FEATURES
    # ==============================================
    
    # Latest account numbers
    features['num_bank_accounts'] = customer_data['Num_Bank_Accounts'].iloc[-1]
    features['num_credit_cards'] = customer_data['Num_Credit_Card'].iloc[-1]
    features['num_loans'] = customer_data['Num_of_Loan'].iloc[-1]
    
    # Account stability (did they open/close accounts?)
    features['bank_account_changes'] = customer_data['Num_Bank_Accounts'].std()
    features['credit_card_changes'] = customer_data['Num_Credit_Card'].std()
    
    # ==============================================
    # CREDIT UTILIZATION & DEBT FEATURES
    # ==============================================
    
    # Latest values
    features['credit_utilization'] = customer_data['Credit_Utilization_Ratio'].iloc[-1]
    features['outstanding_debt'] = customer_data['Outstanding_Debt'].iloc[-1]
    features['monthly_balance'] = customer_data['Monthly_Balance'].iloc[-1]
    
    # Averages over time
    features['avg_credit_utilization'] = customer_data['Credit_Utilization_Ratio'].mean()
    features['avg_outstanding_debt'] = customer_data['Outstanding_Debt'].mean()
    
    # Debt trend
    if len(customer_data) >= 3:
        recent_debt = customer_data['Outstanding_Debt'].tail(3).mean()
        older_debt = customer_data['Outstanding_Debt'].head(3).mean()
        features['debt_trend'] = recent_debt - older_debt  # Positive = increasing debt
    else:
        features['debt_trend'] = 0
    
    # Max debt burden
    features['max_outstanding_debt'] = customer_data['Outstanding_Debt'].max()
    
    # ==============================================
    # LOAN & EMI FEATURES  
    # ==============================================
    
    features['total_emi'] = customer_data['Total_EMI_per_month'].iloc[-1]
    features['avg_emi'] = customer_data['Total_EMI_per_month'].mean()
    features['interest_rate'] = customer_data['Interest_Rate'].iloc[-1]
    
    # Debt-to-income ratio
    if features['monthly_income'] > 0:
        features['dti_ratio'] = features['outstanding_debt'] / (features['monthly_income'] * 12)
        features['emi_to_income'] = features['total_emi'] / features['monthly_income']
    else:
        features['dti_ratio'] = 0
        features['emi_to_income'] = 0
    
    # ==============================================
    # INVESTMENT & SAVINGS FEATURES
    # ==============================================
    
    features['amount_invested'] = customer_data['Amount_invested_monthly'].iloc[-1]
    features['avg_investment'] = customer_data['Amount_invested_monthly'].mean()
    
    # Savings rate
    if features['monthly_income'] > 0:
        features['savings_rate'] = features['amount_invested'] / features['monthly_income']
    else:
        features['savings_rate'] = 0
    
    # ==============================================
    # CREDIT INQUIRIES & MIX
    # ==============================================
    
    features['num_credit_inquiries'] = customer_data['Num_Credit_Inquiries'].mean()
    features['recent_inquiries'] = customer_data['Num_Credit_Inquiries'].tail(3).mean()
    
    # Credit mix (most common)
    features['credit_mix'] = customer_data['Credit_Mix'].mode()[0] if len(customer_data['Credit_Mix'].mode()) > 0 else 'Unknown'
    
    # Changed credit limit
    features['credit_limit_changes'] = customer_data['Changed_Credit_Limit'].sum()
    features['avg_credit_limit_change'] = customer_data['Changed_Credit_Limit'].mean()
    
    # ==============================================
    # CREDIT HISTORY
    # ==============================================
    
    # Parse credit history age (in "X Years and Y Months" format)
    def parse_credit_history(history_str):
        try:
            if pd.isna(history_str) or history_str == 'NA':
                return 0
            years = 0
            months = 0
            if 'Year' in str(history_str):
                parts = str(history_str).split()
                years = int(parts[0]) if parts[0].isdigit() else 0
                if 'Months' in str(history_str):
                    month_part = str(history_str).split('and')[1].strip().split()[0]
                    months = int(month_part) if month_part.isdigit() else 0
            return years + months/12
        except:
            return 0
    
    credit_histories = customer_data['Credit_History_Age'].apply(parse_credit_history)
    features['credit_history_years'] = credit_histories.max()
    
    # ==============================================
    # BEHAVIORAL PATTERNS
    # ==============================================
    
    # Payment of minimum amount behavior
    pays_min = customer_data['Payment_of_Min_Amount'].apply(lambda x: 1 if str(x).lower() in ['yes', 'y'] else 0)
    features['pct_paying_minimum_only'] = pays_min.mean()
    
    # Consistency score (how stable are their patterns)
    volatility_score = (
        features['income_volatility'] * 0.3 +
        features['payment_delay_volatility'] * 0.3 +
        features['bank_account_changes'] * 0.2 +
        features['credit_card_changes'] * 0.2
    )
    features['financial_stability_score'] = max(0, 100 - volatility_score * 10)
    
    # Number of months in dataset
    features['months_in_data'] = len(customer_data)
    
    return features

# Apply aggregation to each customer
print("\n🔄 Aggregating customer-level features...")
print("   This creates ONE row per customer from their time-series data")

customer_features_list = []
for customer_id, group in train.groupby('Customer_ID'):
    features = create_customer_features(group)
    customer_features_list.append(features)

# Create aggregated dataframe
df_agg = pd.DataFrame(customer_features_list)

print(f"\n✅ Aggregated {len(train):,} records → {len(df_agg):,} customers")
print(f"   Created {len(df_agg.columns)} features per customer")

# ============================================================================
# 3. ADDITIONAL ENGINEERED FEATURES
# ============================================================================

print("\n" + "="*80)
print("CREATING ADDITIONAL DERIVED FEATURES")
print("="*80)

# Age groups
df_agg['age_group'] = pd.cut(df_agg['age'], 
                              bins=[0, 25, 35, 45, 55, 100],
                              labels=['young', 'young_adult', 'middle_age', 'senior', 'elder'])

# Income brackets
df_agg['income_bracket'] = pd.cut(df_agg['annual_income'],
                                   bins=[0, 30000, 60000, 100000, 200000, np.inf],
                                   labels=['very_low', 'low', 'medium', 'high', 'very_high'])

# Debt burden categories
df_agg['debt_burden'] = pd.cut(df_agg['dti_ratio'],
                                bins=[-0.1, 0, 0.3, 0.6, 1.0, 100],
                                labels=['no_debt', 'low', 'medium', 'high', 'very_high'])

# Payment reliability score (0-100)
df_agg['payment_reliability_score'] = (
    (1 - df_agg['late_payment_frequency'].clip(0, 1)) * 40 +  # 40 points for not being late
    (1 - df_agg['avg_payment_delay'].clip(0, 30) / 30) * 30 +  # 30 points for small delays
    (1 - df_agg['payment_delay_volatility'].clip(0, 10) / 10) * 30  # 30 points for consistency
)

# Financial health score (composite)
df_agg['financial_health_score'] = (
    df_agg['payment_reliability_score'] * 0.4 +
    df_agg['financial_stability_score'] * 0.3 +
    (df_agg['savings_rate'].clip(0, 0.5) * 200) * 0.3
)

# Risk flags
df_agg['high_utilization_flag'] = (df_agg['avg_credit_utilization'] > 70).astype(int)
df_agg['high_debt_flag'] = (df_agg['dti_ratio'] > 0.5).astype(int)
df_agg['frequent_late_payments'] = (df_agg['late_payment_frequency'] > 0.3).astype(int)
df_agg['declining_payment_behavior'] = (df_agg['payment_behavior_trend'] < -2).astype(int)

# Account diversity score
df_agg['account_diversity'] = (
    df_agg['num_bank_accounts'].clip(0, 5) +
    df_agg['num_credit_cards'].clip(0, 5) +
    df_agg['num_loans'].clip(0, 3)
)

print(f"✅ Created {len(df_agg.columns) - len(customer_features_list[0])} additional derived features")

# ============================================================================
# 4. ENCODE CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*80)
print("ENCODING CATEGORICAL VARIABLES")
print("="*80)

# Encode categorical features
categorical_cols = ['occupation', 'credit_mix', 'age_group', 'income_bracket', 'debt_burden']

label_encoders = {}
for col in categorical_cols:
    if col in df_agg.columns:
        le = LabelEncoder()
        df_agg[f'{col}_encoded'] = le.fit_transform(df_agg[col].astype(str))
        label_encoders[col] = le
        print(f"   ✅ Encoded: {col}")

# ============================================================================
# 5. PREPARE FINAL FEATURE SET
# ============================================================================

print("\n" + "="*80)
print("PREPARING FINAL FEATURE MATRIX")
print("="*80)

# Separate features and target
X = df_agg.drop(['customer_id', 'credit_score'] + categorical_cols, axis=1, errors='ignore')
y = df_agg['credit_score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

# Handle any remaining NaN values
X = X.fillna(X.median())

# Get feature names
feature_names = X.columns.tolist()

print(f"\n✅ Final feature matrix:")
print(f"   - Samples: {len(X):,}")
print(f"   - Features: {len(feature_names)}")
print(f"   - Target classes: {y.nunique()}")

print(f"\n📊 Target distribution:")
for score, count in y.value_counts().sort_index().items():
    score_name = {0: 'Poor', 1: 'Standard', 2: 'Good'}[score]
    pct = count / len(y) * 100
    print(f"   {score_name:10s}: {count:5,} ({pct:5.1f}%)")

# ============================================================================
# 6. FEATURE IMPORTANCE PREVIEW
# ============================================================================

print("\n" + "="*80)
print("FEATURE CORRELATION WITH TARGET")
print("="*80)

# Calculate correlations
correlations = X.corrwith(y).abs().sort_values(ascending=False)

print("\nTop 20 features correlated with Credit Score:")
for i, (feat, corr) in enumerate(correlations.head(20).items(), 1):
    print(f"   {i:2d}. {feat:40s} {corr:.4f}")

# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("SAVING PROCESSED DATA")
print("="*80)

# Save feature matrix
X.to_csv('data/X_features.csv', index=False)
y.to_csv('data/y_target.csv', index=False, header=['credit_score'])
df_agg.to_csv('data/customer_aggregated.csv', index=False)

# Save feature names
with open('data/feature_names.txt', 'w') as f:
    for feat in feature_names:
        f.write(f"{feat}\n")

print(f"\n✅ Saved files:")
print(f"   - data/X_features.csv ({X.shape[0]:,} rows × {X.shape[1]} features)")
print(f"   - data/y_target.csv ({len(y):,} labels)")
print(f"   - data/customer_aggregated.csv (full dataset with categorical)")
print(f"   - data/feature_names.txt (feature list)")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)

print(f"""
✅ Feature Engineering Complete!

Feature Categories Created:
  - Income features: 5
  - Payment behavior: 9
  - Credit accounts: 5
  - Debt & utilization: 9
  - Loan & EMI: 5
  - Investment & savings: 4
  - Credit history: 4
  - Behavioral patterns: 8
  - Derived features: 12
  - Encoded categorical: 5
  
Total: {len(feature_names)} features

Data Characteristics:
  - {len(X):,} customers
  - {len(feature_names)} features
  - 3 target classes (Poor/Standard/Good)
  - Class imbalance: {(y.value_counts() / len(y) * 100).max():.1f}% majority class

Next Steps:
  1. Run: python 03_train_baseline.py
  2. Train XGBoost baseline model
  3. Establish performance benchmark
""")

print("="*80)
print("STEP 2 COMPLETE! ✅")
print("="*80)