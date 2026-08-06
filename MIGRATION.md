# Migrating to DATADOC's fitted pipeline API

Version 0.3 keeps the original `DATADOC` facade for compatibility, but new projects should use `DataDocPipeline`.

```python
from datadoc import DataDocPipeline, PipelineConfig

pipeline = DataDocPipeline(PipelineConfig(target="label"))
pipeline.fit(train_df)
validation_features = pipeline.transform(validation_df)
pipeline.save("pipeline.json")
```

`DATADOC.engineer()` fits and transforms the same file and is therefore only suitable for exploratory cleaning. It emits a deprecation warning for supervised work. Use `fit()` on a training split and `transform()` on validation, test, and inference data instead.

The legacy AI code-execution path has been removed. AI planning, when installed with `datadoc-cli[ai]`, is limited to registered transformations and never executes provider-generated Python.
