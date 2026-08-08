import random
import csv
import uuid

def generate_dirty_data(filepath, rows=1000):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['customer_id', 'age', 'income', 'signup_date', 'plan_type', 'churn'])
        
        for _ in range(rows):
            # Target (0 or 1)
            churn = random.choice([0, 1])
            
            # Identifier (High Cardinality)
            customer_id = str(uuid.uuid4())
            
            # Numeric with missing values and outliers
            age = random.randint(18, 75)
            if random.random() < 0.05:
                age = ''  # missing
            elif random.random() < 0.05:
                age = 999  # outlier
                
            # Numeric with missing values (Income correlates with churn to give the model something to learn)
            # If churn is 1, income tends to be lower
            if churn == 1:
                income = random.randint(20000, 60000)
            else:
                income = random.randint(40000, 120000)
            
            if random.random() < 0.1:
                income = ''
                
            # Datetime string
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            signup_date = f"2023-{month:02d}-{day:02d} {random.randint(0,23):02d}:00"
            if random.random() < 0.02:
                signup_date = ''
                
            # Categorical with some noise
            plan_type = random.choice(['Basic', 'Pro', 'Enterprise'])
            if random.random() < 0.05:
                plan_type = 'Unknown'
            if random.random() < 0.02:
                plan_type = ''
                
            writer.writerow([customer_id, age, income, signup_date, plan_type, churn])

if __name__ == "__main__":
    generate_dirty_data("scratch/dirty_data.csv", 2000)
    print("Created scratch/dirty_data.csv")
