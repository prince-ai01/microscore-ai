import numpy as np
import pandas as pd

# Let's create fake data for 1,000 small shopkeepers
np.random.seed(42)
records = []

for i in range(1, 1001):
    daily_income = np.random.randint(1000, 8000)      # Earns ₹1,000 to ₹8,000/day
    daily_expense = int(daily_income * np.random.uniform(0.5, 0.9)) # Spends 50% to 90% on stock
    monthly_savings = (daily_income - daily_expense) * 30
    upi_transactions = np.random.randint(15, 120)    # Number of QR scans per day
    
    # Simple rule: If savings are very low, risk of not repaying is high
    if monthly_savings < 8000:
        can_repay = 0  # High risk / Default
    else:
        can_repay = 1  # Safe / Repays loan

    records.append({
        'vendor_id': i,
        'daily_income': daily_income,
        'daily_expense': daily_expense,
        'monthly_savings': monthly_savings,
        'daily_upi_count': upi_transactions,
        'repaid_loan': can_repay
    })

# Save it to an Excel/CSV file
df = pd.DataFrame(records)
df.to_csv('vendor_data.csv', index=False)
print("Done! 'vendor_data.csv' created with 1,000 records.")