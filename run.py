import argparse
import logging
from src.core.registries import get_dataframe_loader, get_converter
from src.processing.pipeline import BreastImageProcessingPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[KAPTIOS-AROB-BREAST-DATASETS]")
    parser.add_argument("--dataset_name", type=str, default='vindr',
                        choices=['vindr', 'cbis', 'inbreast'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--tmp_dir", type=str, default=None)
    args = parser.parse_args()
    parser.set_defaults(augment=True)

    logging_message = "[KAPTIOS-AROB-BREAST-DATASETS]"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )
    
    logging.info(f'Running {args.dataset_name} dataset preparation')
    
    processing_pipeline = BreastImageProcessingPipeline()
    dataframe_loader = get_dataframe_loader(args.dataset_name, args.data_dir)
    dataframes = dataframe_loader.load()
    logging.info(f'Dataframes loaded: {list(dataframes.keys())}')
    
    converter = get_converter(args.dataset_name, processing_pipeline, args.batch_size, args.n_workers, args.tmp_dir)
    converter.run(dataframes, args.out_dir)
    logging.info("Processing completed successfully.")
    logging.info("You can now use the processed dataset for training or evaluation.")
    logging.info("Thank you for using the KAPTIOS-AROB-BREAST-DATASETS processing pipeline!")
    logging.info("For any issues or feedback, please refer to the project's documentation or contact the maintainers.")
    logging.info("Exiting the script.")
    logging.info("Goodbye!")
    logging.shutdown()