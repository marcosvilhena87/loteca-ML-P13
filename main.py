from scripts.preprocess_data import preprocess
from scripts.train_model import train
from scripts.predict_results import predict
from scripts.telemetry import init_run, setup_logging


def main() -> None:
    init_run()
    setup_logging()
    preprocess()
    train()
    predict()


if __name__ == "__main__":
    main()
