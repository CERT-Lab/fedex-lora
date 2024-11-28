import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from data_utils import *
from models import *
import argparse
import warnings
from sklearn.metrics import matthews_corrcoef
import numpy as np
import wandb
from train_eval import *
from fed_agg import *
import warnings
import json

parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument("--agg_type", type=str, default="ours", help="Type of aggregation")
parser.add_argument("--rounds", type=int, default=6, help="Number of rounds")
parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
parser.add_argument(
    "--local_epochs", type=int, default=3, help="Number of local epochs"
)
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--lora_r", type=int, default=4, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value")
parser.add_argument(
    "--lora_dropout", type=float, default=0.1, help="LoRA dropout value"
)
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument(
    "--max_seq_length", type=int, default=128, help="Maximum sequence length"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
parser.add_argument("--idx", type=int, default=0, help="Index of the save folder")
parser.add_argument("--log", action="store_true", help="Log the results")
parser.add_argument("--run_dir", type=str, help="Directory to store logs")

args = parser.parse_args()

wandb.init(project="project_name", config=args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

warnings.filterwarnings("ignore")


def get_next_run_number(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1

    existing_runs = [int(d) for d in os.listdir(base_dir) if d.isdigit()]
    return max(existing_runs, default=0) + 1


def save_args(args, directory):
    args_file = os.path.join(directory, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)


def federated_learning(task):

    train_data, val_data, test_data, tokenizer = create_e2e_data()
    client_data = create_client_dataloaders_nlg(train_data, args)

    if args.agg_type == "ffa":
        global_model = create_peft_gpt2_model_e2e_ffa(args)
    else:
        global_model = create_peft_gpt2_model_e2e(args)

    for round in range(args.rounds):
        print(f"Round {round + 1}/{args.rounds}")

        # Train on selected clients
        client_models = []
        for client in range(args.num_clients):

            if args.agg_type == "ffa":
                client_model = create_peft_gpt2_model_e2e_ffa(args)
            else:
                client_model = create_peft_gpt2_model_e2e(args)

            client_model.load_state_dict(global_model.state_dict())
            client_model = train_client_e2e(
                client_model, client_data[client], val_data, tokenizer, args
            )
            client_models.append(client_model)

        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_models)
        elif args.agg_type == "ours":
            global_model = aggregate_models_ours(global_model, client_models, args)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)

        args.idx = round + 1

        if args.log:
            base_dir = "text_store_new/" + args.agg_type
            run_number = get_next_run_number(base_dir)
            run_dir = os.path.join(base_dir, str(run_number))
            os.makedirs(run_dir)
            save_args(args, run_dir)
            args.run_dir = run_dir

            evaluate_e2e_save_text(global_model, test_data, tokenizer, args)

    return global_model


# Main execution
if __name__ == "__main__":
    task = "e2e"
    model = federated_learning(task)
