import os
import sys
import pandas as pd

def run_quick_feature_engineering(input_path=None):
    print("🧹 Running Automated Feature Engineering Pipeline...")
    
    # Use absolute path based on script location
    if input_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_path = os.path.join(project_root, "data", "raw", "Fraud_Data.csv")
    
    # 1. Load the raw baseline dataset
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Baseline data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find file at {input_path}. Please check your path.")
        sys.exit(1)

    # 2. Convert raw text columns to high-precision dates
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # 3. Feature 1: Account Lifespan Maturity (Converted to total seconds)
    print("⏳ Engineering 'time_since_signup' metric...")
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

    # 4. Feature 2: Shifting Transaction Velocity (10-minute lookback)
    print("🚀 Engineering rolling 'tx_count_last_10min' metrics...")
    # Sort chronologically by purchase time to calculate moving velocity windows accurately
    df = df.sort_values(by='purchase_time').reset_index(drop=True)
    
    # Track velocity windows efficiently by comparing timestamp indices per user
    df['tx_count_last_10min'] = 1  # Set default baseline count
    
    # Group by user tracking profiles and calculate rolling time windows
    for user, group in df.groupby('user_id'):
        if len(group) > 1:
            for idx in group.index:
                current_time = df.loc[idx, 'purchase_time']
                ten_minutes_ago = current_time - pd.Timedelta(minutes=10)
                # Count lookback entries inside the 10-minute threshold box
                match_count = group[(group['purchase_time'] >= ten_minutes_ago) & 
                                    (group['purchase_time'] <= current_time)].shape[0]
                df.loc[idx, 'tx_count_last_10min'] = match_count

    # 5. Save the complete dataset back to disk
    df.to_csv(input_path, index=False)
    print(f"🎉 SUCCESS! Dataset fully upgraded and written back to disk. New Shape: {df.shape}")
    print("Columns now available:", list(df.columns))

if __name__ == "__main__":
    run_quick_feature_engineering()