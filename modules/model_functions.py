from tqdm import tqdm_notebook, trange
import torch, os, math, sys
import numpy as np
from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import precision_score, f1_score, recall_score, log_loss, accuracy_score
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from tensorboardX import SummaryWriter

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


######################### TRAINING #########################


def train(train_dataloader, model, args, eval_dataloader=None):
    """
    Trains the model. The params dict contains all parameters for training.

    :param train_dataloader: torch.utils.data.dataloader.DataLoader
    :param model: pytorch_transformers model
    :param args: dict with the model parameters

    Returns: model, train_loss_set, global_step, tr_loss / global_step
    """

    tb_writer = SummaryWriter()

    t_total = args["train_example_length"] // args["gradient_accumulation_steps"] * args["num_train_epochs"]

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() \
                    if not any(nd in n for nd in no_decay)], "weight_decay": args["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() \
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    warmup_steps = math.ceil(t_total * args["warmup_ratio"])
    args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args["warmup_steps"], t_total=t_total)

    if args["fp16"]:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", args['train_example_length'])
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_loss_set = []

    for _ in trange(int(args["num_train_epochs"]), desc="Epoch"):
        for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(args['device']) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args["model_type"] in ["bert", "xlnet"] else None,
                      # XLM don't use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            train_loss_set.append(loss)
            print("\r%f" % loss, end='')

            if args["gradient_accumulation_steps"] > 1:
                loss = loss / args["gradient_accumulation_steps"]

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])

            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

            tr_loss += loss.item()
            if (step + 1) % args["gradient_accumulation_steps"] == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args["logging_steps"] == 0:
                    # Log metrics
                    if args["evaluate_during_training"]:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(model, eval_dataloader, args)
                #                        for key, value in results.items():
                #                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                #                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                #                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args["logging_steps"], global_step)
                #                    logging_loss = tr_loss

                if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args["output_dir"], "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

    return model, train_loss_set, global_step, tr_loss / global_step


######################### EVALUATION #########################

def get_eval_report(preds, predictions, labels):
    """
    Returns the evaluation report

    :param preds: np array of probas
    :param predictions: np array of predicted labels
    :param labels: np array of labels

    Returns: dict with results.
    """

    assert len(preds) == len(labels)

    mcc = matthews_corrcoef(labels, predictions)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    logloss = log_loss(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    return {
        "mcc": mcc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1-score": f1,
        "logloss": logloss,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def evaluate(model, eval_dataloader, args, prefix=""):
    """
    Evaluates the model. The model parameters are in the args dict.
    The eval_dataloader contains all evaluation example in the form of InputFeatures
    while the all_label_ids contains their label.

    :param model: pytorch model
    :param eval_dataloader: list of InputFeatures
    :param args: dict, contains all model parameters
    :param prefix: str

    Returns dict with results.
    """

    eval_output_dir = args["output_dir"]
    results = {}
    eval_task = args["task_name"]

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args["device"]) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps

    if args["output_mode"] == "classification":
        predictions = np.argmax(preds, axis=1)
    elif args["output_mode"] == "regression":
        predictions = np.squeeze(preds)

    results = get_eval_report(preds, predictions, out_label_ids)
    results.update(results)
    results["task_name"] = args["task_name"]
    results["model"] = args["model"]
    results.update(results)
    results["eval_loss"] = eval_loss

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
    return results


def main():
    params = torch.load(sys.argv[1])

    from pytorch_transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,
                                      XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                                      XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                      RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }

    config_class, model_class, tokenizer_class = MODEL_CLASSES[params['model_type']]
    model = model_class.from_pretrained(params['model'])
    model.to(params["device"])

    data_path = params['data_dir']
    max_seq_length = params['max_seq_len']

    if params["evaluate_during_training"]:
        val_dataloader_name = f'{data_path}val_dataloader_{max_seq_length}_{params["model_type"]}'
        print(f'Load evaluation during training dataloader from {val_dataloader_name}...')
        val_dataloader = torch.load(val_dataloader_name)

    if params['do_train']:
        dataloader_name = f'{data_path}dataloader_{max_seq_length}_{params["model_type"]}'
        print(f'Load train dataloader saved in {dataloader_name}..')
        train_dataloader = torch.load(dataloader_name)

    if params['do_train'] and params['evaluate_during_training']:
        print(f'Start training... Will evaluate ')
        model, train_loss_set, global_step, tr_loss = train(train_dataloader, model, params, val_dataloader)

    elif params['do_train']:
        print(f'Start training...')
        model, train_loss_set, global_step, tr_loss = train(train_dataloader, model, params)

    if "weights_name" not in params:
        params["weights_name"] = "pytorch_model.bin"
    if "config_name" not in params:
        params["config_name"] = "config.json"

    output_model_file = os.path.join(params["output_dir"], params["weights_name"])
    output_config_file = os.path.join(params["output_dir"], params["config_name"])

    print(f'Save model, config and vocab to {params["output_dir"]}..')
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(params["output_dir"])
    print(f'Model, config and vocab are saved in {params["output_dir"]}..')

    if params["do_eval"]:
        eval_dataloader_name = f'{data_path}eval_dataloader_{max_seq_length}_{params["model_type"]}'
        print(f'Load evaluation dataloader from {eval_dataloader_name}...')
        eval_dataloader = torch.load(eval_dataloader_name)
        print(f'Evaluation dataloader is loaded.')

        result = evaluate(model, eval_dataloader, params, prefix="")
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        output_results_file = os.path.join(params["output_dir"], "results.json")
        print(f'Saving results to {output_results_file}.')
        torch.save(result, output_results_file)

        return
    else:
        return


if __name__ == "__main__":
    main()


