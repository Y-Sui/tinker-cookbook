import json

from datasets import load_dataset

# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("zai-org/LongWriter-6k", "default", split="train")
# print(ds["messages"][0][0]["content"])

# # save to a local jsonl file
# with open(
#     "/ndata/yuansui/tinker-cookbook/tinker_cookbook/data/longwriter_6k_sample.jsonl", "w"
# ) as f:
#     for item in ds["messages"]:
#         # f.write(json.dumps(item, ensure_ascii=False) + "\n")
#         query = item[0]["content"]
#         f.write(json.dumps({"query": query}) + "\n")


# ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
# with open("/ndata/yuansui/tinker-cookbook/tinker_cookbook/data/math500_sample.jsonl", "w") as f:
#     for item in ds:
#         f.write(
#             json.dumps(
#                 {"query": item["problem"], "answer": item["answer"], "solution": item["solution"]}
#             )
#             + "\n"
#         )


ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
with open("/ndata/yuansui/tinker-cookbook/tinker_cookbook/data/aime2024_sample.jsonl", "w") as f:
    for item in ds:
        f.write(
            json.dumps(
                {"query": item["problem"], "answer": item["answer"], "solution": item["solution"]}
            )
            + "\n"
        )

ds = load_dataset("MathArena/aime_2025", split="train")
with open("/ndata/yuansui/tinker-cookbook/tinker_cookbook/data/aime2025_sample.jsonl", "w") as f:
    for item in ds:
        f.write(json.dumps({"query": item["problem"], "answer": item["answer"]}) + "\n")


# ds = load_dataset("quehry/HelloBench", split="test")
# with open("/ndata/yuansui/tinker-cookbook/tinker_cookbook/data/hellobench_sample.jsonl", "w") as f:
#     for item in ds:
#         f.write(
#             json.dumps(
#                 {
#                     "query": item["instruction"],
#                     "category": item["category"],
#                     "formatted_checklists": item["formatted_checklists"],
#                 }
#             )
#             + "\n"
#         )


# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
# with open(
#     "/ndata/yuansui/tinker-cookbook/tinker_cookbook/data/gpqa_diamond_sample.jsonl", "w"
# ) as f:
#     for item in ds:
#         instruction = item["Question"]

#         # 1. Collect all choices into a list
#         choices = [
#             item["Correct Answer"],
#             item["Incorrect Answer 1"],
#             item["Incorrect Answer 2"],
#             item["Incorrect Answer 3"],
#         ]

#         # Remove any empty strings just in case
#         choices = [c for c in choices if c and str(c).strip()]

#         # 2. SHUFFLE the choices so A is not always the correct answer
#         random.shuffle(choices)

#         # 3. Format options (A., B., ...) and identify the correct formatted string
#         formatted_options = []
#         final_answer_string = ""

#         for i, choice in enumerate(choices):
#             letter = chr(65 + i)  # 0->A, 1->B, 2->C, etc.
#             option_str = f"{letter}. {choice}"
#             formatted_options.append(option_str)

#             # Check if this specific choice is the correct original answer
#             if choice == item["Correct Answer"]:
#                 final_answer_string = option_str

#         # Join the options with newlines
#         options_text = "\n".join(formatted_options)

#         # 4. Construct the query
#         query = instruction + "\nOptions:\n" + options_text

#         # 5. Write to file
#         f.write(
#             json.dumps(
#                 {
#                     "query": query,
#                     "answer": final_answer_string,  # This will now be "A. Correct Answer Text"
#                     "explanation": item["Explanation"],
#                 }
#             )
#             + "\n"
#         )
