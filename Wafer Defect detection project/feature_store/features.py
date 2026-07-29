"""Feature View definitions for the Feast feature store."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

dataset = Entity(
    name="dataset",
    join_keys=["dataset_id"],
    value_type=ValueType.INT64,  
    description="Unique ID for each dataset version",
)

dataset_source = FileSource(
    path="s3://datasets/metadata/dataset_metadata.parquet",
    timestamp_field="event_timestamp",
)

dataset_metadata = FeatureView(
    name="dataset_metadata",
    entities=[dataset],
    ttl=timedelta(days=365),
    schema=[
        Field(name="num_train_samples", dtype=Int64),
        Field(name="num_test_samples", dtype=Int64),
        Field(name="num_val_samples", dtype=Int64),
        Field(name="num_classes", dtype=Int64),
        Field(name="image_height", dtype=Int64),
        Field(name="image_width", dtype=Int64),
        Field(name="pixel_min", dtype=Float32),
        Field(name="pixel_max", dtype=Float32),
        Field(name="pixel_mean", dtype=Float32),
        Field(name="pixel_std", dtype=Float32),
        Field(name="avg_labels_per_sample", dtype=Float32),
        Field(name="max_labels_per_sample", dtype=Int64),
        Field(name="num_duplicates_train", dtype=Int64),
    ],
    source=dataset_source,
    online=True,
)