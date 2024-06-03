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

def convert_to_jsonl(ds_name, ds_dir, jsonl_name, limit=0, start_at=0, splits=None):
    if ds_name == 'ReasonVQA':
        convert_to_jsonl_reasonvqa(ds_dir, jsonl_name, limit, start_at, splits)
    elif ds_name == 'VQAv2':
        convert_to_jsonl_vqa(ds_dir, jsonl_name, limit, start_at)
    elif ds_name == 'OKVQA':
        convert_to_jsonl_vqa(ds_dir, jsonl_name, limit, start_at, okvqa=True)
    else:
        raise Exception('Invalid dataset name')


def convert_to_jsonl_reasonvqa(ds_dir, jsonl_name, limit=0, start_at=0, splits=None):
    if splits is None:
        splits = ['train', 'test']

    with jsonlines.open(jsonl_name, 'w') as jsonl_file:
        i = 0
        for split in splits:
            json_data = stream_data(f'{ds_dir}/{split}.json', limit=limit, start_at=start_at)
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


def convert_to_jsonl_vqa(ds_dir, jsonl_name, limit=0, start_at=0, okvqa=False):
    if okvqa:
        question_file = f'{ds_dir}/OpenEnded_mscoco_val2014_questions.json'
        annotation_file = f'{ds_dir}/mscoco_val2014_annotations.json'
    else:
        question_file = f'{ds_dir}/v2_OpenEnded_mscoco_val2014_questions.json'
        annotation_file = f'{ds_dir}/v2_mscoco_val2014_annotations.json'

    questions = {}
    with open(question_file) as f:
        datareader = ijson.items(f, 'questions.item')
        for d in datareader:  # image_id, question, question_id
            questions[d['question_id']] = d['question']

    all_answers = []
    with open(annotation_file) as f:
        datareader = ijson.items(f, 'annotations.item')
        for d in datareader:
            if okvqa:
                ans = set([a['answer'] for a in d['answers']])
            else:
                ans = [d['multiple_choice_answer']]
            for a in ans:
                if a not in all_answers:
                    all_answers.append(a)

    print('Num of answers:', len(all_answers))

    jsonl_file = jsonlines.open(jsonl_name, 'w')

    i = 0
    with open(annotation_file) as f:
        datareader = ijson.items(f, 'annotations.item')
        # {"question_type": "none of the above", "multiple_choice_answer": "down", "answers": [{"answer": "down", "answer_confidence": "yes", "answer_id": 1}, {"answer": "down", "answer_confidence": "yes", "answer_id": 2}, {"answer": "at table", "answer_confidence": "yes", "answer_id": 3}, {"answer": "skateboard", "answer_confidence": "yes", "answer_id": 4}, {"answer": "down", "answer_confidence": "yes", "answer_id": 5}, {"answer": "table", "answer_confidence": "yes", "answer_id": 6}, {"answer": "down", "answer_confidence": "yes", "answer_id": 7}, {"answer": "down", "answer_confidence": "yes", "answer_id": 8}, {"answer": "down", "answer_confidence": "yes", "answer_id": 9}, {"answer": "down", "answer_confidence": "yes", "answer_id": 10}], "image_id": 262148, "answer_type": "other", "question_id": 262148000}
        for d in datareader:
            i += 1

            if i == 1 or i % 100 == 0:
                print(f"[{i}]: {d['image_id']}")

            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            if okvqa:
                answers = list(set([a['answer'] for a in d['answers']]))
            else:
                answers = [d['multiple_choice_answer']]

            text = d['question'] + '\n'

            choices = [*answers]
            choices += select_choices(all_answers, answers)
            choice_scores = [1] * len(answers) + [0] * (len(choices) - len(answers))

            shuffled_choices, shuffled_choice_scores = shuffle(choices, choice_scores)
            for ii in range(len(shuffled_choices)):
                text += f'{chr(ii + 65)}. {shuffled_choices[ii]}\n'

            text += "Answer with the option's letter from the given choices directly."

            jsonl_file.write({
                "question_id": d['question_id'],
                "image": f"COCO_val2014_{str(d['image_id']).zfill(12)}.jpg",
                "text": text,
                "answers": [chr(ii + 65) for ii in range(len(shuffled_choice_scores)) if
                            shuffled_choice_scores[ii] == 1]
            })

    jsonl_file.close()


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


# json file is either "train.json" or "test.json"
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


def select_choices(answers, true_answers, k=3):
    if len(answers) <= k:
        return answers

    n = len(answers)
    choices = []
    while len(choices) < k:
        idx = random.randint(0, n - 1)
        if answers[idx] not in true_answers and answers[idx] not in choices:
            choices.append(answers[idx])
    return choices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, default='ReasonVQA', help='Valid input: ReasonVQA, VQAv2, OKVQA, GQA')
    parser.add_argument('--ds_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--jsonl_name', type=str, required=True, help='Name of jsonl output')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    args = parser.parse_args()

    convert_to_jsonl(args.ds_name, args.ds_dir, args.jsonl_name, limit=args.limit, start_at=args.start_at)
