import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
n_samples = 1000

# Define symptoms (features)
symptoms = ['ho', 'sot', 'dau_hong', 'dau_bung', 'mat_khuu_giac', 'kho_tho']

# Define diseases (labels)
diseases = ['cam_cum', 'covid_19', 'viem_hong', 'roi_loan_tieu_hoa', 'binh_thuong']

# Generate synthetic data
data = []

for _ in range(n_samples):
    # Randomly generate symptoms
    row = np.random.randint(0, 2, size=len(symptoms))
    
    # Logic to assign disease based on symptoms
    if row[0] == 1 and row[1] == 1 and row[2] == 1:  # Ho, sốt, đau họng
        if row[4] == 1 or row[5] == 1:  # Mất khứu giác hoặc khó thở
            disease = 'covid_19'  # COVID-19
        else:
            disease = 'cam_cum'  # Cảm cúm
    elif row[0] == 1 and row[2] == 1 and row[1] == 0:  # Ho, đau họng, không sốt
        disease = 'viem_hong'  # Viêm họng
    elif row[3] == 1:  # Đau bụng
        disease = 'roi_loan_tieu_hoa'  # Rối loạn tiêu hóa
    elif np.sum(row) == 0 or np.sum(row) == 1:  # Không có triệu chứng hoặc chỉ có 1 triệu chứng nhẹ
        disease = 'binh_thuong'  # Bình thường
    else:
        # Randomly assign for cases that don't fit the rules above
        probs = [0.3, 0.2, 0.2, 0.2, 0.1]
        disease = np.random.choice(diseases, p=probs)
    
    # Add to dataset
    data.append(list(row) + [disease])

# Create DataFrame
columns = symptoms + ['disease']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
output_file = os.path.join(os.path.dirname(__file__), 'synthetic_health_data.csv')
df.to_csv(output_file, index=False)

print(f"Generated {n_samples} samples and saved to {output_file}")
print(f"Disease distribution:\n{df['disease'].value_counts()}") 