import polars as pl
from datadoc.core.pipeline import DataDocPipeline, PipelineConfig

def main():
    print("Testing DATADOC SDK programmatically...")
    
    # 1. Load data
    print("Loading data...")
    df = pl.read_csv("scratch/dirty_data.csv")
    
    # 2. Init Pipeline
    print("Initializing pipeline...")
    pipeline = DataDocPipeline(PipelineConfig(
        target="churn",
        drop_identifiers=True,
        scaling="standard"
    ))
    
    # 3. Profile
    print("Profiling data...")
    profile = pipeline.profile(df)
    print(f"Profile found {profile.columns} columns.")
    
    # 4. Plan
    print("Planning transformations...")
    plan = pipeline.plan(df)
    print(f"Plan generated {len(plan.operations)} operations.")
    
    # 5. Fit
    print("Fitting pipeline (learning medians, bounds, vocabularies)...")
    pipeline.fit(df)
    
    # 6. Transform
    print("Transforming data...")
    clean_df = pipeline.transform(df)
    
    # Save output
    output_path = "scratch/clean_data.csv"
    if isinstance(clean_df, pl.LazyFrame):
        clean_df.collect().write_csv(output_path)
    else:
        clean_df.write_csv(output_path)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    main()
