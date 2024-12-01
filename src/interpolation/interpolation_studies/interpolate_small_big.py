from transformers import AutoTokenizer, GPTNeoXForCausalLM
import eval_utils
import argparse
from src.interpolation.interpolate import Interpolate
import json
import matplotlib.pyplot as plt
from src.interpolation.grow_hyper import grow_hyperclone
import shutil

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--small_pretrained_model_name", type=str, required=True, default="EleutherAI/pythia-70m"
    )

    parser.add_argument(
        "--large_pretrained_model_name", type=str, required=True, default="EleutherAI/pythia-410m"
    )

    parser.add_argument(
        '--tasks', type=str, nargs="+", required=True, default=["lambada_openai", "paloma_dolma-v1_5"]
    )

    parser.add_argument(
        '--task_columns', type=str, nargs="+", default=["perplexity,none", "word_perplexity,none"], help="The columns to use for the task evaluation"
    )

    parser.add_argument(
        '--num_interpolations', type=int, default=5
    )

    parser.add_argument(
        '--output_json', type=str, default="eval/interpolation_small_big.json"
    )

    parser.add_argument(
        '--output_plot', type=str, default="plots/interpolation_small_big"
    )

    parser.add_argument(
        '--large_depth', type=int, default=24, help='Desired depth of the large model'
    )

    parser.add_argument(
        '--large_width', type=int, default=1024, help='Desired width of the large model'
    )
    
    parser.add_argument(
        '--depth_growth', type=str, default='alternate', choices=['alternate', 'append'], help='Depth growth strategy'
    )

    parser.add_argument(
        '--skip_eval', action='store_true', help='Skip evaluation'
    )

    return parser.parse_args()


def main():

    args = parse_args()


    # Print memory usage
    print("Memory usage after loading large pretrained model")
    eval_utils.print_memory_usage()

    # Grow small model
    print("Growing small model")
    small_model = GPTNeoXForCausalLM.from_pretrained(args.small_pretrained_model_name)

    grown_small_model, copied_layers = grow_hyperclone.grow_model(
        model=small_model,
        new_depth=args.large_depth,
        new_width=args.large_width,
        depth_growth=args.depth_growth,
    )

    # Save grown_small_model to file
    grown_small_model.save_pretrained("models/grown_small_model_temp")

    # Interpolate with ratios
    alphas, results = Interpolate.interpolate_evaluate_models_llm(
        model_1_path=args.large_pretrained_model_name,
        model_2_path="models/grown_small_model_temp",
        n_interpolations=args.num_interpolations,
        callback=eval_utils.evaluate_model_llm_eval,
        checkpoint=None,
        tasks=args.tasks,
        tokenizer_path=args.small_pretrained_model_name,
        num_fewshot=0,
        batch_size=4,
        parallelize=True,
        skip_eval=args.skip_eval,
    )

    # Delete the grown_small_model
    shutil.rmtree("models/grown_small_model_temp")

    # Save results to file
    if not args.skip_eval:
        with open(args.output_json, "w") as f:
            json.dump(results, f)
        print(f"Interpolation results saved to {args.output_json}")

    else:
        results = json.load(open(args.output_json, "r"))

    # Plot the word_perplexity,none values for each alpha
    for i, task in enumerate(args.tasks):
        print(f"Plotting task {task}")
        plt.figure()
        plt.plot(alphas, [v[task][args.task_columns[i]] for k, v in results.items()], marker='o', linestyle='-', color='b')
        plt.xlabel("Alpha (fraction of large model)")
        plt.ylabel(args.task_columns[i])
        plt.grid(True)
        plt.title(f"Interpolation between small and large models for task {task}")
        plt.savefig(args.output_plot + f"_{task}.png")




if __name__ == '__main__':

    main()