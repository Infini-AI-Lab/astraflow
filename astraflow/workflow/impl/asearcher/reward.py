"""Reward and format helpers for the ASearcher workflow."""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def bool_mapping(text: str) -> str:
    if text == "True":
        return "yes"
    if text == "False":
        return "no"
    return text


def contains_chinese(text: str) -> bool:
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
        if "\u3400" <= char <= "\u4dbf":
            return True
        if "\uf900" <= char <= "\ufaff":
            return True
    return False


def extract_solution(solution_str: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def em_check(prediction: str, golden_answers: str | list[str]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    for golden_answer in golden_answers:
        if normalize_answer(bool_mapping(golden_answer)) == normalized_prediction:
            return 1
    return 0


def normalize_text(text: str) -> str:
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def f1_score(answer_content: str, ground_truth: str) -> float:
    answer_content = normalize_text(bool_mapping(answer_content))
    ground_truth = normalize_text(bool_mapping(ground_truth))

    if contains_chinese(ground_truth):
        def parse_chinese_str(value: str) -> set[str]:
            numbers: list[str] = []
            for idx, char in enumerate(value):
                if char.isdigit():
                    if idx > 0 and value[idx - 1].isdigit():
                        numbers[-1] = numbers[-1] + char
                    else:
                        numbers.append(char)
            for char in "0123456789，。 ,.-":
                value = value.replace(char, "")
            return set(list(value) + numbers)

        pred_tokens = parse_chinese_str(answer_content)
        gt_tokens = parse_chinese_str(ground_truth)
    else:
        pred_tokens = set(answer_content.split())
        gt_tokens = set(ground_truth.split())

    if not gt_tokens or not pred_tokens:
        return 0.0

    common_tokens = pred_tokens & gt_tokens
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_score_em(
    solution_str: str,
    ground_truth: str | list[str],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> tuple[str | None, float]:
    del method
    if isinstance(ground_truth, list):
        answer = extract_solution(solution_str)
        return answer, max(compute_score_em(solution_str, gt)[1] for gt in ground_truth)

    answer = extract_solution(solution_str)
    if answer is None:
        return None, 0.0
    if em_check(answer, ground_truth):
        return answer, score
    return answer, format_score


def compute_score_f1(
    solution_str: str,
    ground_truth: str | list[str],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> tuple[str | None, float]:
    del method, format_score, score
    if isinstance(ground_truth, list):
        answer = extract_solution(solution_str)
        return answer, max(compute_score_f1(solution_str, gt)[1] for gt in ground_truth)

    answer = extract_solution(solution_str)
    if answer is None:
        return None, 0.0
    return answer, f1_score(answer, ground_truth)


def correct_format_fn(idx: int, text: str) -> bool:
    del idx
    return all(
        [
            text.count("<search>") == text.count("</search>"),
            text.count("<access>") == text.count("</access>"),
            text.count("<answer>") == text.count("</answer>"),
            text.count("<search>") + text.count("<access>") + text.count("<answer>") <= 1,
            text.count("Assistant") == text.count("assistant") == 0,
            text.count("</think>") <= 1,
        ]
    )
