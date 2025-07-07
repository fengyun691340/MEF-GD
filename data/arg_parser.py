import os
import argparse


def eval_parse_args() -> argparse.Namespace:
    """ This function parses the arguments passed to the script.

    Returns:
        argparse.Namespace: Namespace containing the arguments.
    """
    
    parser = argparse.ArgumentParser(description="Multimodal Garment Designer argparse.")

    # Diffusion parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
       
    # dataset parameters
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument("--category", type=str, default="", help="category to use")
    parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"],
                        help="Test order, should be either paired or unpaired")


    args = parser.parse_args()

    # if not, set default local rank
    #env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    #if env_local_rank != -1 and env_local_rank != args.local_rank:
    #    args.local_rank = env_local_rank

    return args
