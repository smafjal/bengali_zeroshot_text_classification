__author__ = "smafjal"
__email__ = "afjal.sm19@gmail.com"
__date__ = "2/17/21-12:44"

import csv
import json
import logging
import os
import random

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)

from tqdm import tqdm, trange
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
from transformers.optimization import AdamW

from configs import get_args
from hypothesis import get_hypothesis, get_topics
from utils import evaluate_Yahoo_zeroshot_TwpPhasePred, BnTokenizer

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("zsbn-train")

hypothesis = get_hypothesis()
topics = get_topics()
bntokinizer = BnTokenizer()


def load_model(ckpt):
    return torch.load(ckpt)


def save_model(model, ckpt):
    torch.save(model, ckpt)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MMCJsonDataProcessor(DataProcessor):
    def __init__(self):
        super(MMCJsonDataProcessor, self).__init__()

    @staticmethod
    def get_train_examples(filename, size_limit_per_type=-1):
        with open(filename) as r:
            data = json.load(r)
        logger.info(f"Train | data len: {len(data)}")

        line_co = 0
        exam_co = 0
        examples = []
        topics_types = set(topics)

        for sample in data:
            text = sample.get('text')
            text = " ".join(bntokinizer.word_tokenize(text.strip()))
            type_list = set(sample.get('topic_wise').keys())
            neg_types = topics_types - set(type_list)

            if len(neg_types) > 3:
                sampled_type_set = random.sample(neg_types, 3)
            else:
                continue
            # pos pair
            text_a = text
            for topic in type_list:
                for hypo in hypothesis.get(topic):
                    guid = "train-" + str(exam_co)
                    text_b = hypo
                    label = 'entailment'
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                    exam_co += 1
            # neg pair
            for topic in sampled_type_set:
                for hypo in hypothesis.get(topic):
                    guid = "train-" + str(exam_co)
                    text_b = hypo
                    label = 'not_entailment'  # if line[0] == '1' else 'not_entailment'
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                    exam_co += 1
            if exam_co > size_limit_per_type != -1:
                break
            line_co += 1
            if line_co % 1000 == 0:
                logger.info('loading train line: {}'.format(line_co))
        logger.info("Total train examples: {}".format(len(examples)))
        return examples, topics_types

    @staticmethod
    def get_dev_examples(filename, seen_types, size_limit_per_type=-1):
        with open(filename) as r:
            data = json.load(r)
        logger.info(f"Test | data len: {len(data)}")

        line_co = 0
        exam_co = 0
        examples = []

        hypo_seen_str_indicator = []
        hypo_2_type_index = []
        for topic in topics:
            hypo_list = hypothesis.get(topic)
            for hypo in hypo_list:
                hypo_2_type_index.append(topic)  # this hypo is for type i
                if topic in seen_types:
                    hypo_seen_str_indicator.append('seen')  # this hypo is for a seen type
                else:
                    hypo_seen_str_indicator.append('unseen')

        gold_label_list = []
        for sample in data:
            text = sample.get('text')
            text = " ".join(bntokinizer.word_tokenize(text.strip()))

            type_index = list(sample.get('topic_wise').keys())
            gold_label_list.append(type_index)

            for topic, hypo_list in hypothesis.items():
                if topic in type_index:
                    # pos pair
                    for hypo in hypo_list:
                        guid = "test-" + str(exam_co)
                        text_a = text
                        text_b = hypo
                        label = 'entailment'
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                        exam_co += 1
                else:
                    # neg pair
                    for hypo in hypo_list:
                        guid = "test-" + str(exam_co)
                        text_a = text
                        text_b = hypo
                        label = 'not_entailment'
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                        exam_co += 1
            if exam_co > size_limit_per_type != -1:
                break

            line_co += 1
            if line_co % 200 == 0:
                logger.info('loading dev line: {}'.format(line_co))
            # if line_co == 20:
            #     break

        logger.info("Total test examples: {}".format(len(examples)))
        return examples, gold_label_list, hypo_seen_str_indicator, hypo_2_type_index

    def get_labels(self):
        return ["entailment", "not_entailment"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features_tk(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples, 1):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        text_a = example.text_a
        text_b = example.text_b
        tokens_ids = tokenizer(
            text_a, text_b,
            return_tensors='pt',
            padding='max_length',
            max_length=max_seq_length,
            truncation=True
        )
        label_id = label_map[example.label]
        features.append(
            InputFeatures(
                input_ids=tokens_ids['input_ids'].squeeze(0).tolist(),
                input_mask=tokens_ids['attention_mask'].squeeze(0).tolist(),
                segment_ids=tokens_ids['token_type_ids'].squeeze(0).tolist() if "token_type_ids" in tokens_ids else
                None,
                label_id=label_id
            )
        )
    return features


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    premise_2_tokenzed = {}
    hypothesis_2_tokenzed = {}
    list_2_tokenizedID = {}

    features = []
    for (ex_index, example) in enumerate(examples, 1):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = premise_2_tokenzed.get(example.text_a)
        if tokens_a is None:
            tokens_a = tokenizer.tokenize(example.text_a)
            premise_2_tokenzed[example.text_a] = tokens_a

        tokens_b = premise_2_tokenzed.get(example.text_b)
        if tokens_b is None:
            tokens_b = tokenizer.tokenize(example.text_b)
            hypothesis_2_tokenzed[example.text_b] = tokens_b

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens_A = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_A = [0] * len(tokens_A)
        tokens_B = tokens_b + ["[SEP]"]
        segment_ids_B = [1] * len(tokens_B)
        tokens = tokens_A + tokens_B
        segment_ids = segment_ids_A + segment_ids_B

        input_ids_A = list_2_tokenizedID.get(' '.join(tokens_A))
        if input_ids_A is None:
            input_ids_A = tokenizer.convert_tokens_to_ids(tokens_A)
            list_2_tokenizedID[' '.join(tokens_A)] = input_ids_A
        input_ids_B = list_2_tokenizedID.get(' '.join(tokens_B))
        if input_ids_B is None:
            input_ids_B = tokenizer.convert_tokens_to_ids(tokens_B)
            list_2_tokenizedID[' '.join(tokens_B)] = input_ids_B
        input_ids = input_ids_A + input_ids_B

        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index == 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id
            ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'F1':
        return {"f1": f1_score(y_true=labels, y_pred=preds)}
    else:
        raise KeyError(task_name)


def fit_model(args, processor):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        # device = torch.device('cpu')
        # n_gpu = 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    label_list = processor.get_labels()
    num_labels = len(label_list)

    pretrain_model_dir = args.bert_model
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir)

    # tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir)
    # tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)

    logger.info("Train File: {}".format(args.train_file))
    logger.info("Valid File: {}".format(args.valid_file))

    train_limit = 100
    test_limit = 50

    train_examples, seen_types = processor.get_train_examples(args.train_file, train_limit)
    # train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
    train_features = convert_examples_to_features_tk(train_examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # load dev set
    eval_examples, eval_label_list, eval_hypo_seen_str_indicator, eval_hypo_2_type_index = \
        processor.get_dev_examples(args.valid_file, seen_types, test_limit)
    # eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
    eval_features = convert_examples_to_features_tk(eval_examples, label_list, args.max_seq_length, tokenizer)

    eval_all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(eval_all_input_ids, eval_all_input_mask, eval_all_segment_ids, eval_all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # --------------

    # load test set
    test_examples, test_label_list, test_hypo_seen_str_indicator, test_hypo_2_type_index = \
        processor.get_dev_examples(args.test_file, seen_types, test_limit)
    
    # test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
    test_features = convert_examples_to_features_tk(test_examples, label_list, args.max_seq_length, tokenizer)
    test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    # --------------

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(
        str(PYTORCH_TRANSFORMERS_CACHE), 'distributed_{}'.format(args.local_rank)
    )

    model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=len(label_list))
    # model = BertForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=len(label_list))
    # model = RobertaForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=len(label_list))

    if args.fp16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_unseen_acc = 0.0
    max_dev_unseen_acc = 0.0
    max_dev_seen_acc = 0.0
    max_overall_acc = 0.0

    ckpt = os.path.join(args.output_dir, pretrain_model_dir.split('/')[-1])
    os.makedirs(ckpt, exist_ok=True)

    iter_co = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # logits = model(input_ids, segment_ids, input_mask, labels=None)
            logits = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=None
            )

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits[0].view(-1, num_labels), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            iter_co += 1

        save_path = os.path.join(ckpt, "model_epoch_{}.pt".format(iter_co))

        # Evalute model after training
        model.eval()
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        print('Evaluating...')
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            logits = logits[0]

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        """
        preds: size*2 (entail, not_entail) wenpeng added a softxmax so that each row is a prob vec
        """
        pred_probs = softmax(preds, axis=1)[:, 0]
        pred_binary_labels_harsh = []
        pred_binary_labels_loose = []
        for i in range(preds.shape[0]):
            if preds[i][0] > preds[i][1] + 0.1:
                pred_binary_labels_harsh.append(0)
            else:
                pred_binary_labels_harsh.append(1)
            if preds[i][0] > preds[i][1]:
                pred_binary_labels_loose.append(0)
            else:
                pred_binary_labels_loose.append(1)

        seen_acc, unseen_acc = evaluate_Yahoo_zeroshot_TwpPhasePred(
            pred_probs, pred_binary_labels_harsh,
            pred_binary_labels_loose,
            eval_label_list,
            eval_hypo_seen_str_indicator,
            eval_hypo_2_type_index, seen_types
        )

        if unseen_acc > max_dev_unseen_acc:
            max_dev_unseen_acc = unseen_acc
        print(
            '\ndev seen_acc & acc_unseen:', seen_acc, unseen_acc,
            ' max_dev_unseen_acc:', max_dev_unseen_acc, '\n'
        )

        # start evaluate on test set after this epoch
        model.eval()
        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_loss = 0
        nb_test_steps = 0
        preds = []
        print('Testing...')
        for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            logits = logits[0]
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        # eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        pred_probs = softmax(preds, axis=1)[:, 0]
        pred_binary_labels_harsh = []
        pred_binary_labels_loose = []
        for i in range(preds.shape[0]):
            if preds[i][0] > preds[i][1] + 0.1:
                pred_binary_labels_harsh.append(0)
            else:
                pred_binary_labels_harsh.append(1)
            if preds[i][0] > preds[i][1]:
                pred_binary_labels_loose.append(0)
            else:
                pred_binary_labels_loose.append(1)

        seen_acc, unseen_acc = evaluate_Yahoo_zeroshot_TwpPhasePred(
            pred_probs, pred_binary_labels_harsh,
            pred_binary_labels_loose,
            test_label_list,
            test_hypo_seen_str_indicator,
            test_hypo_2_type_index, seen_types
        )
        # result = compute_metrics('F1', preds, all_label_ids.numpy())
        # loss = tr_loss/nb_tr_steps if args.do_train else None
        # test_acc = mean_f1#result.get("f1")
        if unseen_acc > max_test_unseen_acc:
            max_test_unseen_acc = unseen_acc
        print(
            '\n\n\t test seen_acc & acc_unseen:', seen_acc, unseen_acc,
            ' max_test_unseen_acc:', max_test_unseen_acc, '\n'
        )


def main():
    args = get_args()
    processor = MMCJsonDataProcessor()
    pretrain_model_dir = args.bert_model
    logger.info("Bert Model Name: {}".format(pretrain_model_dir))

    if args.do_train:
        model = fit_model(
            args=args,
            processor=processor
        )
    else:
        ckpt = os.path.join(args.output_dir, "v1/senti_baseline_multi_bn.pt")
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir)
        model = load_model(ckpt)


if __name__ == "__main__":
    main()
