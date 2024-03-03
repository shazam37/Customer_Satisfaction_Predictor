from pipeline.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="data/olist_customers_dataset.csv")
    