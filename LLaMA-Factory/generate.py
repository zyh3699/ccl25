import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto


def generate_responses(model_path, data, output_file):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for item in data:
        prompt = item["instruction"]
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        item["response"] = response

    # Save results to the specified output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Responses generated and saved to {output_file}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate responses using a specified model and input data.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/qwen2_lora_sft",
        help="Path to the pre-trained model (default: output/qwen2_lora_sft)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="factivity_test_data.json",
        help="Path to the JSON data file containing instructions (default: factivity_test_data.json)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.json",
        help="Path to save the output JSON file with generated responses (default: output.json)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the data file
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Generate responses and save results
    generate_responses(args.model_path, data, args.output_file)