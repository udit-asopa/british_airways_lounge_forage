from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------------
# TASK 1: Lounge Eligibility Modeling
# -----------------------------------

def process_lounge_schedule(schedule_path: Path, output_path: Path):
    print("Processing lounge eligibility data...")

    df = pd.read_csv(schedule_path)
    df['DepartureDateTime'] = pd.to_datetime(df['FLIGHT_DATE'] + ' ' + df['FLIGHT_TIME'], errors='coerce')

    def categorize_time_of_day(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Midday"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"

    df['TimeCategory'] = df['DepartureDateTime'].dt.hour.map(categorize_time_of_day)

    # Lounge eligibility assumptions
    lookup_table = {
        ('Morning', 'Long Haul', 'North America'): {'Tier1': 0.10, 'Tier2': 0.20, 'Tier3': 0.50},
        ('Morning', 'Long Haul', 'Europe'):        {'Tier1': 0.05, 'Tier2': 0.15, 'Tier3': 0.40},
        ('Midday', 'Short Haul', 'Europe'):        {'Tier1': 0.01, 'Tier2': 0.05, 'Tier3': 0.30},
        ('Evening', 'Long Haul', 'Asia'):          {'Tier1': 0.08, 'Tier2': 0.18, 'Tier3': 0.45},
        ('Night', 'Long Haul', 'North America'):   {'Tier1': 0.12, 'Tier2': 0.22, 'Tier3': 0.55},
        'default':                                 {'Tier1': 0.03, 'Tier2': 0.10, 'Tier3': 0.35}
    }

    def get_assumptions(row):
        return lookup_table.get((row['TimeCategory'], row['HAUL'], row['ARRIVAL_REGION']), lookup_table['default'])

    def estimate_eligibility(row):
        assumptions = get_assumptions(row)
        total = row['FIRST_CLASS_SEATS'] + row['BUSINESS_CLASS_SEATS'] + row['ECONOMY_SEATS']
        return pd.Series({
            'Est_Tier1_PAX': assumptions['Tier1'] * total,
            'Est_Tier2_PAX': assumptions['Tier2'] * total,
            'Est_Tier3_PAX': assumptions['Tier3'] * total,
        })

    eligibility_df = df.apply(estimate_eligibility, axis=1)
    df = pd.concat([df, eligibility_df], axis=1)

    summary = (
        df.groupby(['TimeCategory', 'HAUL', 'ARRIVAL_REGION'])[['Est_Tier1_PAX', 'Est_Tier2_PAX', 'Est_Tier3_PAX']]
        .sum().round().astype(int).reset_index()
    )

    summary.to_csv(output_path, index=False)
    print(f"Lounge eligibility summary saved to: {output_path}")


# --------------------------------
# TASK 2: Booking Prediction Model
# --------------------------------

def train_booking_model(data_path: Path, model_output_path: Path, feature_plot_path: Path):
    print("\n Training predictive model for holiday bookings...")

    df = pd.read_csv(data_path, encoding='latin1')

    # Label Encoding
    categorical_cols = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Features and labels
    X = df.drop(columns=['booking_complete'])
    y = df['booking_complete']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_output_path)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\nCross-Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature Importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:\n", importance_df)
    importance_df.to_csv(model_output_path.with_suffix('_features.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance from Random Forest')
    plt.tight_layout()
    plt.savefig(feature_plot_path)
    plt.close()
    print(f"Feature importance plot saved to: {feature_plot_path}")

    print("\nKey Slide Takeaways:")
    print("- Top predictors:", ', '.join(importance_df['Feature'].head(5)))
    print(f"- CV Accuracy: {cv_scores.mean():.3f}")


if __name__ == "__main__":
    base_path = Path(__file__).parent

    # Task 1: Lounge Eligibility
    process_lounge_schedule(
        schedule_path=base_path / "data" / "british_airways_schedule.csv",
        output_path=base_path / "data" / "lounge_eligibility_summary.csv"
    )

    # Task 2: Predictive Booking Model
    train_booking_model(
        data_path=base_path / "data" / "customer_booking.csv",
        model_output_path=base_path / "booking_predictor_model.pkl",
        feature_plot_path=base_path / "data" / "feature_importance.png"
    )
