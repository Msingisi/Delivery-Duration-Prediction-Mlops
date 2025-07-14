import logging
from pipelines.training_pipeline import delivery_duration_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """
    Entry point to run the delivery duration training pipeline.
    """
    logger.info("Starting the Delivery Duration Training Pipeline...")
    delivery_duration_pipeline()
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()