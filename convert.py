import argparse

import ijson
import jsonlines
import random


# {
# "question_id": 101669,
# "prompt": "How many towels are in the image?\nA. One\nB. Two\nC. Three\nD. Four\nAnswer with the option's letter from the given choices directly.",
# "text": "B",
# "answer_id": "nR4KfzvqJQadAAwFjwn9bv",
# "model_id": "llava-v1.5-13b",
# "metadata": {}
# }

def convert_to_jsonl(path_to_dataset, output_jsonl_name='jsonl', limit=0, start_at=0, splits=None):
    if splits is None:
        splits = ['train', 'test']

    with jsonlines.open(output_jsonl_name, 'w') as jsonl_file:
        i = 0
        for split in splits:
            json_data = stream_data(f'{path_to_dataset}/{split}.json', limit=limit, start_at=start_at)
            for d in json_data:
                i += 1

                if i == 1 or i % 100 == 0:
                    print(f"[{i}]: {d['image_id']}")

                local_img_path = f"{split}/{d['image_id']}.jpg"
                text = d['question'] + '\n'

                shuffled_choices, shuffled_choice_scores = shuffle(d['choices'], d['choice_scores'])
                for ii in range(len(shuffled_choices)):
                    text += f'{chr(ii + 65)}. {shuffled_choices[ii]}\n'

                text += "Answer with the option's letter from the given choices directly."

                jsonl_file.write({
                    "question_id": d['question_id'],
                    "image": local_img_path,
                    "text": text,
                    "answers": [chr(ii + 65) for ii in range(len(shuffled_choice_scores)) if
                                shuffled_choice_scores[ii] == 1]
                })


'''
n_questions: int
exported_time: datetime
questions: array
    image_id
    image_name
    image_dir
    dataset_name
    question_id
    question
    answers
    answers_scores
    choices
    choice_scores
    property_id
    property_label
    n_hop
    has_scene_graph
'''


def stream_data(path_to_json_file, limit=0, start_at=0):
    i = 0
    with open(path_to_json_file) as f:
        datareader = ijson.items(f, 'questions.item')
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            yield record


def shuffle(choices, choice_scores):
    n = len(choices)
    for i in range(n):
        j = random.randint(0, n - 1)
        if i != j:
            tmp1 = choices[i]
            tmp2 = choice_scores[i]
            choices[i] = choices[j]
            choice_scores[i] = choice_scores[j]
            choices[j] = tmp1
            choice_scores[j] = tmp2

    return choices, choice_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ds', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_jsonl_name', type=str, default='output', help='Path to output')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    args = parser.parse_args()

    convert_to_jsonl(args.path_to_ds, args.output_jsonl_name, limit=args.limit, start_at=args.start_at)
