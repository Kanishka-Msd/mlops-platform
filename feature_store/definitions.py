from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64

person = Entity(
    name="person",
    join_keys=["person_id"],
    value_type=None,
    description="A person in the income dataset"
)

personal_source = FileSource(
    path="/Users/kanish/mlops-platform/feature_store/data/personal_features.parquet",
    timestamp_field="event_timestamp"
)

work_source = FileSource(
    path="/Users/kanish/mlops-platform/feature_store/data/work_features.parquet",
    timestamp_field="event_timestamp"
)

financial_source = FileSource(
    path="/Users/kanish/mlops-platform/feature_store/data/financial_features.parquet",
    timestamp_field="event_timestamp"
)

personal_fv = FeatureView(
    name="personal_features",
    entities=[person],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age",            dtype=Int64),
        Field(name="race",           dtype=Int64),
        Field(name="sex",            dtype=Int64),
        Field(name="native-country", dtype=Int64),
    ],
    source=personal_source,
    description="Personal demographic features"
)

work_fv = FeatureView(
    name="work_features",
    entities=[person],
    ttl=timedelta(days=365),
    schema=[
        Field(name="workclass",      dtype=Int64),
        Field(name="education",      dtype=Int64),
        Field(name="education-num",  dtype=Int64),
        Field(name="occupation",     dtype=Int64),
        Field(name="hours-per-week", dtype=Int64),
    ],
    source=work_source,
    description="Work and education features"
)

financial_fv = FeatureView(
    name="financial_features",
    entities=[person],
    ttl=timedelta(days=365),
    schema=[
        Field(name="fnlwgt",         dtype=Int64),
        Field(name="capital-gain",   dtype=Int64),
        Field(name="capital-loss",   dtype=Int64),
        Field(name="marital-status", dtype=Int64),
        Field(name="relationship",   dtype=Int64),
    ],
    source=financial_source,
    description="Financial and family features"
)
