__author__ = "smafjal"
__email__ = "afjal.sm19@gmail.com"
__date__ = "2/17/21-12:52"

import re
import string


class BnTokenizer:
    SENTENCE_END_CHARS = ['\|', '\|\|', '\u0964', '\u0965']
    SENTENCE_END_CHARS_REGEX = '|'.join(SENTENCE_END_CHARS)
    PATTERN = rf'(?<=[{SENTENCE_END_CHARS_REGEX}])\s'

    @staticmethod
    def word_tokenize(text):
        mod_punc = string.punctuation.replace("|", "")
        punc_pattern = re.compile('([' + mod_punc + '\u0964\u0965' + ']|\|+)')
        tok_str = punc_pattern.sub(r' \1 ', text.replace('\t', ' '))
        return re.sub(r'[ ]+', u' ', tok_str).strip(' ').split(' ')

    @staticmethod
    def document_tokenize(text):
        assert isinstance(text, str)
        return re.split(BnTokenizer.PATTERN, text)


def evaluate_Yahoo_zeroshot_TwpPhasePred(
        pred_probs, pred_binary_labels_harsh, pred_binary_labels_loose, eval_label_list,
        eval_hypo_seen_str_indicator, eval_hypo_2_type_index, seen_types
):
    """
    pred_probs: a list, the prob for  "entail"
    pred_binary_labels: a lit, each  for 0 or 1
    eval_label_list: the gold type index; list length == lines in dev.txt
    eval_hypo_seen_str_indicator: totally hypo size, seen or unseen
    eval_hypo_2_type_index:: total hypo size, the type in [0,...n]
    seen_types: a set of type indices
    """

    pred_probs = list(pred_probs)
    # pred_binary_labels = list(pred_binary_labels)
    total_hypo_size = len(eval_hypo_seen_str_indicator)
    total_premise_size = len(eval_label_list)
    assert len(pred_probs) == total_premise_size * total_hypo_size
    assert len(eval_hypo_seen_str_indicator) == len(eval_hypo_2_type_index)

    # print('seen_types:', seen_types)
    # print('eval_hypo_seen_str_indicator:', eval_hypo_seen_str_indicator)
    # print('eval_hypo_2_type_index:', eval_hypo_2_type_index)

    seen_hit = 0
    unseen_hit = 0
    seen_size = 0
    unseen_size = 0

    for i in range(total_premise_size):
        pred_probs_per_premise = pred_probs[i * total_hypo_size: (i + 1) * total_hypo_size]
        pred_binary_labels_per_premise_harsh = pred_binary_labels_harsh[
                                               i * total_hypo_size: (i + 1) * total_hypo_size]
        pred_binary_labels_per_premise_loose = pred_binary_labels_loose[
                                               i * total_hypo_size: (i + 1) * total_hypo_size]

        # print('pred_probs_per_premise:',pred_probs_per_premise)
        # print('pred_binary_labels_per_premise:', pred_binary_labels_per_premise)

        '''first check if seen types get 'entailment'''
        seen_get_entail_flag = False
        for j in range(total_hypo_size):
            if eval_hypo_seen_str_indicator[j] == 'seen' and \
                    pred_binary_labels_per_premise_loose[j] == 0:
                seen_get_entail_flag = True
                break
        '''first check if unseen types get 'entailment'''
        unseen_get_entail_flag = False
        for j in range(total_hypo_size):
            if eval_hypo_seen_str_indicator[j] == 'unseen' and \
                    pred_binary_labels_per_premise_loose[j] == 0:
                unseen_get_entail_flag = True
                break

        if seen_get_entail_flag and unseen_get_entail_flag or \
                (not seen_get_entail_flag and not unseen_get_entail_flag):
            '''compare their max prob'''
            max_prob_seen = -1.0
            max_seen_index = -1
            max_prob_unseen = -1.0
            max_unseen_index = -1
            for j in range(total_hypo_size):
                its_prob = pred_probs_per_premise[j]
                if eval_hypo_seen_str_indicator[j] == 'unseen':
                    if its_prob > max_prob_unseen:
                        max_prob_unseen = its_prob
                        max_unseen_index = j
                else:
                    if its_prob > max_prob_seen:
                        max_prob_seen = its_prob
                        max_seen_index = j
            if max_prob_seen - max_prob_unseen > 0.1:
                pred_type = eval_hypo_2_type_index[max_seen_index]
            else:
                pred_type = eval_hypo_2_type_index[max_unseen_index]

        elif unseen_get_entail_flag:
            '''find the unseen type with highest prob'''
            max_j = -1
            max_prob = -1.0
            for j in range(total_hypo_size):
                if eval_hypo_seen_str_indicator[j] == 'unseen':
                    its_prob = pred_probs_per_premise[j]
                    if its_prob > max_prob:
                        max_prob = its_prob
                        max_j = j
            pred_type = eval_hypo_2_type_index[max_j]

        elif seen_get_entail_flag:
            '''find the seen type with highest prob'''
            max_j = -1
            max_prob = -1.0
            for j in range(total_hypo_size):
                if eval_hypo_seen_str_indicator[j] == 'seen' and \
                        pred_binary_labels_per_premise_loose[j] == 0:
                    its_prob = pred_probs_per_premise[j]
                    if its_prob > max_prob:
                        max_prob = its_prob
                        max_j = j
            assert max_prob > 0.5
            pred_type = eval_hypo_2_type_index[max_j]
        gold_type = eval_label_list[i]

        # print('pred_type:', pred_type, 'gold_type:', gold_type)
        if gold_type in seen_types:
            seen_size += 1
            if gold_type == pred_type:
                seen_hit += 1
        else:
            unseen_size += 1
            if gold_type == pred_type:
                unseen_hit += 1

    seen_acc = seen_hit / (1e-6 + seen_size)
    unseen_acc = unseen_hit / (1e-6 + unseen_size)

    return seen_acc, unseen_acc
