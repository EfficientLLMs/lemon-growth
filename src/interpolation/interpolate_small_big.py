from transformers import AutoTokenizer, GPTNeoXForCausalLM
import eval_utils
import argparse
from interpolate import Interpolate
import json
import matplotlib.pyplot as plt

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
        '--task_column', type=str, default="word_perplexity,none", help="The column to use for the task evaluation"
    )

    parser.add_argument(
        '--num_interpolations', type=int, default=5
    )

    parser.add_argument(
        '--output_json', type=str, default="eval/interpolation_small_big.json"
    )

    parser.add_argument(
        '--output_plot', type=str, default="plots/interpolation_small_big.png"
    )

    return parser.parse_args()


def main():

    args = parse_args()


    # Print memory usage
    print("Memory usage after loading large pretrained model")
    eval_utils.print_memory_usage()

    # Interpolate with ratios
    alphas, results = Interpolate.interpolate_evaluate_models_llm(
        model_1_path=args.large_pretrained_model_name,
        model_2_path=args.small_pretrained_model_name,
        n_interpolations=args.num_interpolations,
        callback=eval_utils.evaluate_model_llm_eval,
        kwargs={
            "checkpoint": None,
            "tasks": args.tasks,
            "tokenizer_path": None,
            "num_fewshot": 0,
            "batch_size": 4,
            "parallelize": True
        }
    )

    # Save results to file
    with open(args.output_json, "w") as f:
        json.dump(results, f)
    print(f"Interpolation results saved to {args.output_json}")

    # Plot the word_perplexity,none values for each alpha
    plt.figure()
    plt.plot(alphas, [r[args.task_column] for r in results])
    plt.xlabel("Alpha (fraction of large model)")
    plt.ylabel(args.task_column)
    plt.title("Interpolation between small and large models")
    plt.savefig(args.output_plot)

    print(f'Interpolation results saved to {args.output_json}')



if __name__ == '__main__':

    main()