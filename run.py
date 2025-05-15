import argparse
import logging
from src.core.df_loaders.registry import get_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[KAPTIOS-AROB-BREAST-DATASETS]")
    parser.add_argument("--dataset_name", type=str, default='vindr',
                        choices=['vindr', 'cbis', 'inbreast'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--augment", action='store_true')
    args = parser.parse_args()
    parser.set_defaults(augment=True)

    logging_message = "[KAPTIOS-AROB-BREAST-DATASETS]"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )

    logging.info(f'Running {args.dataset_name} dataset preparation')
    dataframes = get_datasets(args.dataset_name, args.data_dir)
    print(dataframes)