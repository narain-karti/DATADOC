"""Generate a realistic demo dataset for testing DATADOC CLI."""
import random, csv, uuid
from datetime import datetime, timedelta

random.seed(42)
rows = 500
departments = ["Engineering", "Sales", "Marketing", "HR", "Support"]
cities = ["Chennai", "Bangalore", "Mumbai", "Delhi", "Hyderabad"]

with open("demo.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["customer_id", "name", "age", "salary", "department", "city",
                 "years_experience", "satisfaction_score", "signup_date", "churn"])
    for i in range(rows):
        age = random.randint(22, 60) if random.random() > 0.05 else None          # 5% missing
        salary = random.randint(25000, 150000) if random.random() > 0.04 else None # 4% missing
        dept = random.choice(departments) if random.random() > 0.03 else None      # 3% missing
        city = random.choice(cities)
        yrs = random.randint(0, 30) if random.random() > 0.02 else None
        score = round(random.uniform(1, 10), 1) if random.random() > 0.06 else None
        signup = (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))).strftime("%Y-%m-%d")
        churn = random.choice([0, 1])

        # Inject a few extreme outliers
        if random.random() < 0.02:
            salary = random.randint(900000, 1500000)
        if random.random() < 0.02:
            age = random.randint(150, 999)

        w.writerow([str(uuid.uuid4()), f"Person_{i}", age, salary, dept, city, yrs, score, signup, churn])

print("Created demo.csv with 500 rows.")
