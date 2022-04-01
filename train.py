import torch
import argparse
import logging
import json
import time
import numpy as np
import random
import math
from itertools import chain
from torch.optim import Adam
import gc
import config
from pykp.io import SEP_WORD, EOS_WORD
from Model import *
import torch.nn.functional as F
from utils.time_log import time_since
from utils.data_loader import *
from utils.utils import *
import torch.nn as nn
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
import sys
import os

EPS = 1e-6


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
    if opt.gpuid != 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuid)
        opt.gpuid = 0
    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if opt.delimiter_type == 0:
        opt.delimiter_word = SEP_WORD
    else:
        opt.delimiter_word = EOS_WORD

    # my configuration
    opt.data = "processed_data/{}/".format(opt.data_tag)
    opt.vocab = opt.data
    opt.exp = 'trial.' + opt.data_tag if opt.trial else opt.data_tag

    # seq2seq setting
    if 'Weibo' in opt.data_tag:
        opt.vocab_size = 50000
        opt.word_vec_size = 150
    elif 'Twitter' in opt.data_tag:
        opt.vocab_size = 30000
        opt.word_vec_size = 150
    elif 'StackExchange' in opt.data_tag:
        opt.vocab_size = 50000
        opt.word_vec_size = 150
    elif 'kp20k' in opt.data_tag:
        opt.vocab_size = 50000
        opt.word_vec_size = 150
    else:
        print('Wrong data_tag!!')
        return
    opt.encoder_size = 150
    opt.decoder_size = 300
    size_tag = ".emb{}".format(opt.word_vec_size) + ".vs{}".format(opt.vocab_size) + ".dec{}".format(opt.decoder_size)

    # only train ntm
    if opt.only_train_ntm:
        assert opt.ntm_warm_up_epochs > 0 and not opt.load_pretrain_ntm
        opt.exp += '.topic_num{}'.format(opt.topic_num)
        opt.exp += '.ntm_warm_up_%d' % opt.ntm_warm_up_epochs
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)
        print("Only training the ntm for %d epochs and save it to %s" % (opt.ntm_warm_up_epochs, opt.model_path))
        return opt

    # joint train settings
    if opt.joint_train:
        opt.exp += '.joint_train'
        if opt.add_two_loss:
            opt.exp += '.add_two_loss'
        if opt.joint_train_strategy != 'p_1_joint':
            opt.exp += '.' + opt.joint_train_strategy
            opt.p_seq2seq_e = int(opt.joint_train_strategy.split('_')[1])
            if opt.joint_train_strategy.split('_')[-1] != 'joint':
                opt.iterate_train_ntm = True

    # adding topic settings
    if opt.bridge != "copy":
        opt.exp += '.{}_bridge'.format(opt.bridge)
    if opt.copy_attention:
        opt.exp += '.copy'
    opt.exp += '.seed{}'.format(opt.seed)
    opt.exp += '.topic_num{}'.format(opt.topic_num)
    opt.exp += size_tag
    # fill time into the name
    if opt.model_path.find('%s') > 0:
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    logging.info('Model_PATH : ' + opt.model_path)
    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, 'initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, 'initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(os.path.join(opt.model_path, 'initial.json'), 'w'))

    return opt


def init_optimizers(model, ntm_model, opt):
    optimizer_seq2seq = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate)
    whole_params = chain(model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)
    return optimizer_seq2seq, optimizer_ntm, optimizer_whole


def test_ntm_one_epoch(model, dataloader, opt, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(opt.device)
            data_bow_norm = F.normalize(data_bow)
            if data_bow_norm.shape[0] != opt.batch_size:
                break
            recon_loss, kld, theta, depend1, depend2, beta1, beta2, beta3 = model(data_bow_norm, 1)
            depend1 = F.softmax(depend1 / (0.05 / (0.1 + 1e-5)), 1)
            depend2 = F.softmax(depend2, 1)
            theta1, theta2, theta3 = theta[0], theta[1], theta[2]
            effective_dims_bool = np.ones(opt.topic_num - opt.n_topic2 - 1) > 0
            tree = build_tree(depend2, depend1, beta1, beta2, beta3, [effective_dims_bool])
            theta = torch.cat([theta1, theta2, theta3], dim=1)
            print_tree(tree, opt.bow_dictionary)
            loss = recon_loss + kld
            loss = torch.sum(loss)
            test_loss += loss.item()
    avg_loss = test_loss / len(dataloader.dataset)
    logging.info('====> Test epoch: {} Average loss:  {:.4f}'.format(epoch, avg_loss))
    return avg_loss


def test_model(data_loader, model, ntm_model, opt):
    model.eval()
    ntm_model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    print("Evaluate loss for %d batches" % len(data_loader))
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            if not opt.one2many:  # load one2one dataset
                src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, src_bow = batch
            else:  # load one2many dataset
                src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = batch
                num_trgs = [len(trg_str_list) for trg_str_list in
                            trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size
            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
            batch_size = src.size(0)
            if batch_size != opt.batch_size:
                break
            n_batch += batch_size
            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)
            src_bow = src_bow.to(opt.device)
            src_bow_norm = F.normalize(src_bow)
            recon_loss, kld, theta1, depend1, depend2, beta1, beta2, beta3 = ntm_model(src_bow_norm, 1)
            ntm_loss = recon_loss + kld
            ntm_loss = torch.sum(ntm_loss)
            beta = torch.cat([beta1, beta2, beta3], 0)
            start_time = time.time()
            decoder_dist, h_t, attention_dist, encoder_final_state, coverage, kld_loss, _, _ \
                = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, beta)
            forward_time = time_since(start_time)
            forward_time_total += forward_time
            start_time = time.time()
            if opt.copy_attention:  # Compute the loss using target with oov words
                loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                            coverage_loss=False)
            else:  # Compute the loss using target without oov words
                loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                            coverage_loss=False)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time
            loss += kld_loss + ntm_loss
            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)
            if (batch_i + 1) % (len(data_loader) // 5) == 0:
                print("Train: %d/%d batches, current avg loss: %.3f" %
                      ((batch_i + 1), len(data_loader), evaluation_loss_sum / total_trg_tokens))
    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def train_one_batch(batch, model, ntm_model, optimizer, opt, batch_i, decay):
    # train for one batch
    src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, src_bow = batch
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)
    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)
    optimizer.zero_grad()
    src_bow = src_bow.to(opt.device)
    src_bow_norm = F.normalize(src_bow)
    recon_loss, kld, theta1, depend1, depend2, beta1, beta2, beta3 = ntm_model(src_bow_norm, decay)
    beta = torch.cat([beta1, beta2, beta3], 0)
    ntm_loss = recon_loss + kld * decay
    ntm_loss = torch.sum(ntm_loss)
    start_time = time.time()
    # for one2one setting
    decoder_dist, h_t, attention_dist, encoder_final_state, coverage, kld_loss, _, _ \
        = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, beta)
    forward_time = time_since(start_time)
    start_time = time.time()
    if opt.copy_attention:  # Compute the loss using target with oov words
        loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, opt.coverage_loss)
    else:  # Compute the loss using target without oov words
        loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, opt.coverage_loss)
    loss_compute_time = time_since(start_time)
    total_trg_tokens = sum(trg_lens)
    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("src")
        print(src)
        print(src_oov)
        print(src_lens)
        print(src_mask)
        print("trg")
        print(trg)
        print(trg_oov)
        print(trg_lens)
        print(trg_mask)
        print("oov list")
        print(oov_lists)
        print("Decoder")
        print(decoder_dist)
        print(h_t)
        print(attention_dist)
        raise ValueError("Loss is NaN")
    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        normalization = src.size(0)
    else:
        raise ValueError('The type of loss normalization is invalid.')
    assert normalization > 0, 'normalization should be a positive number'
    start_time = time.time()
    loss += ntm_loss
    # back propagation on the normalized loss
    loss += decay * kld_loss
    loss.div(normalization).backward()
    backward_time = time_since(start_time)
    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
    optimizer.step()
    # construct a statistic object for the loss
    stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time,
                          loss_compute_time=loss_compute_time, backward_time=backward_time)
    return stat, decoder_dist.detach()


def train_ntm(model, optimizer, opt):
    logging.info('======================  Start Training  =========================')
    print("\nWarming up ntm for %d epochs" % opt.ntm_warm_up_epochs)
    _, valid_bow_loader = load_valid_data(opt)
    train_bow_loader = None
    decay_rate = 0.03
    for epoch in range(1, int(opt.ntm_warm_up_epochs / 10) + 1):
        if train_bow_loader:
            del train_bow_loader
            gc.collect()
        _, train_bow_loader = load_train_data(opt, True, (epoch - 1) % 3)
        model.train()
        train_loss = 0
        for i in range(10):
            dataloader = train_bow_loader
            ratio_list = torch.zeros([opt.topic_num - opt.n_topic2 - 1])
            for batch_idx, data_bow in enumerate(dataloader):
                data_bow = data_bow.to(opt.device)
                # normalize data
                data_bow_norm = F.normalize(data_bow)
                optimizer.zero_grad()
                decay = decay_rate * i
                if data_bow_norm.shape[0] != opt.batch_size:
                    break
                recon_loss, kld, theta, depend, depend2, beta1, beta2, beta3 = model(data_bow_norm, decay)
                depend = F.softmax(depend / (0.05 / (0.1 + 1e-5)), 1)
                depend2 = F.softmax(depend2, 1)
                theta = theta[0]
                theta = theta.to('cpu')
                loss = recon_loss + kld * decay
                loss = torch.sum(loss)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                for s in range(opt.batch_size):
                    ratio_list += theta[s]
                if batch_idx % 100 == 0:
                    ratio_list /= torch.sum(ratio_list)
                    cdf = []
                    cur = 1
                    for s in ratio_list.detach().numpy():
                        cur -= s
                        cdf.append(cur)
                    effective_dims_bool = np.array(cdf) > 0.05
                    effective_dims = np.sum(effective_dims_bool)
                    print('Train Epoch: {}[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        (epoch - 1) * 10 + i + 1, batch_idx * len(data_bow), len(dataloader.dataset),
                        100. * batch_idx / len(dataloader),
                        loss.item() / len(data_bow)))
                    print('effective_dims:', effective_dims)
                    tree = build_tree(depend2, depend, beta1, beta2, beta3, [effective_dims_bool])
                    print_tree(tree, opt.bow_dictionary)
            logging.info('====>Train epoch: {} Average loss: {:.4f}'.format(
                (epoch - 1) * 10 + i + 1, train_loss / len(dataloader.dataset)))
            if (i + 1) % 5 == 0:
                val_loss = test_ntm_one_epoch(model, valid_bow_loader, opt, (epoch - 1) * 10 + i + 1)
                best_ntm_model_path = os.path.join(opt.model_path,
                                                   'e%d.val_loss=%.3f.effective_dims%d.ntm_model' %
                                                   ((epoch - 1) * 10 + i + 1, val_loss, effective_dims))
                logging.info("\nSaving warm up ntm model into %s" % best_ntm_model_path)
                torch.save(model.state_dict(), open(best_ntm_model_path, 'wb'))


def train_model(model, ntm_model, optimizer_ml, optimizer_ntm, optimizer_whole, bow_dictionary, opt):
    def convert_time2str(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%dh-%02dm" % (h, m)

    total_batch = 0
    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    best_ntm_valid_loss = float('inf')
    joint_train_patience = 1
    ntm_train_patience = 1
    global_patience = 5
    num_stop_dropping = 0
    num_stop_dropping_ntm = 0
    num_stop_dropping_global = 0
    t0 = time.time()
    decay_rate = 0.03
    print("\nEntering main training for %d epochs" % opt.epochs)
    valid_data_loader, _ = load_valid_data(opt, load_train=True)
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        optimizer = optimizer_whole
        model.train()
        ntm_model.train()
        logging.info("\nTraining seq2seq+ntm epoch: {}/{}".format(epoch, opt.epochs))
        for i in range(3):
            train_data_loader, _ = load_train_data(opt, load_train=True, i=i)
            logging.info("The total num of batches: %d, current learning rate:%.6f" %
                         (len(train_data_loader), optimizer.param_groups[0]['lr']))
            for batch_i, batch in enumerate(train_data_loader):
                total_batch += 1
                if batch[0].shape[0] != opt.batch_size:
                    break
                batch_loss_stat, _ = train_one_batch(batch, model, ntm_model, optimizer, opt, batch_i,
                                                     decay_rate * epoch)
                report_train_loss_statistics.update(batch_loss_stat)
                total_train_loss_statistics.update(batch_loss_stat)
                if (batch_i + 1) % 100 == 0:
                    print("Train: %d/%d batches epoch%d" %
                          ((batch_i + 1), len(train_data_loader), epoch))
                if (total_batch+1) % 500 == 0:
                    print("Train: %d/%d batches, current avg loss: %.3f" %
                          ((batch_i + 1), len(train_data_loader), batch_loss_stat.xent()))

                    current_train_ppl = report_train_loss_statistics.ppl()
                    current_train_loss = report_train_loss_statistics.xent()
                    # test the model on the validation dataset for one epoch
                    model.eval()
                    valid_loss_stat = test_model(valid_data_loader, model, ntm_model, opt)
                    current_valid_loss = valid_loss_stat.xent()
                    current_valid_ppl = valid_loss_stat.ppl()
                    model.train()
                    ntm_model.train()
                    # debug
                    if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (
                                epoch, batch_i, total_batch))
                        exit()

                    if current_valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
                        print("Valid loss drops")
                        sys.stdout.flush()
                        best_valid_loss = current_valid_loss
                        best_valid_ppl = current_valid_ppl
                        num_stop_dropping = 0
                        num_stop_dropping_global = 0
                        if epoch >= opt.start_checkpoint_at and epoch > opt.p_seq2seq_e and not opt.save_each_epoch:
                            check_pt_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.model-%s' %
                                                               (epoch, current_valid_loss,
                                                                convert_time2str(time.time() - t0)))
                            # save model parameters
                            torch.save(
                                model.state_dict(),
                                open(check_pt_model_path, 'wb')
                            )
                            logging.info('Saving seq2seq checkpoints to %s' % check_pt_model_path)
                            if opt.joint_train:
                                check_pt_ntm_model_path = check_pt_model_path.replace('.model-', '.model_ntm-')
                                # save model parameters
                                torch.save(
                                    ntm_model.state_dict(),
                                    open(check_pt_ntm_model_path, 'wb')
                                )
                                logging.info('Saving ntm checkpoints to %s' % check_pt_ntm_model_path)
                    else:
                        print("Valid loss does not drop")
                        sys.stdout.flush()
                        num_stop_dropping += 1
                        num_stop_dropping_global += 1
                        # decay the learning rate by a factor
                        for i, param_group in enumerate(optimizer.param_groups):
                            old_lr = float(param_group['lr'])
                            new_lr = old_lr * opt.learning_rate_decay
                            if old_lr - new_lr > EPS:
                                param_group['lr'] = new_lr
                                print("The new learning rate for seq2seq is decayed to %.6f" % new_lr)

                    if opt.save_each_epoch:
                        check_pt_model_path = os.path.join(opt.model_path,
                                                           'e%d.train_loss=%.3f.val_loss=%.3f.model-%s' %
                                                           (epoch, current_train_loss, current_valid_loss,
                                                            convert_time2str(time.time() - t0)))
                        torch.save(  # save model parameters
                            model.state_dict(),
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving seq2seq checkpoints to %s' % check_pt_model_path)

                        if opt.joint_train:
                            check_pt_ntm_model_path = check_pt_model_path.replace('.model-', '.model_ntm-')
                            torch.save(  # save model parameters
                                ntm_model.state_dict(),
                                open(check_pt_ntm_model_path, 'wb')
                            )
                            logging.info('Saving ntm checkpoints to %s' % check_pt_ntm_model_path)
                    # log loss, ppl, and time
                    logging.info('Epoch: %d; Time spent: %.2f' % (epoch, time.time() - t0))
                    logging.info(
                        'avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                            current_train_ppl, current_valid_ppl, best_valid_ppl))
                    logging.info(
                        'avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                            current_train_loss, current_valid_loss, best_valid_loss))

                    report_train_ppl.append(current_train_ppl)
                    report_valid_ppl.append(current_valid_ppl)
                    report_train_loss.append(current_train_loss)
                    report_valid_loss.append(current_valid_loss)
                    report_train_loss_statistics.clear()
                    if not opt.save_each_epoch and num_stop_dropping >= opt.early_stop_tolerance:  # not opt.joint_train or
                        logging.info(
                            'Have not increased for %d check points, early stop training' % num_stop_dropping)
                        break
    return


def main(opt):
    try:
        word2idx, idx2word, vocab, bow_dictionary = load_vocab(opt)
        opt.bow_vocab_size = len(bow_dictionary)
        start_time = time.time()
        ntm_model = nTSNTM(1, 10, opt.bow_vocab_size, 256, opt.topic_num - opt.n_topic2 - 1, opt.n_topic2,
                           opt.learning_rate, opt.batch_size, "sigmoid", opt.device).to(opt.device)
        model = Seq2SeqModel(opt).to(opt.device)
        optimizer_seq2seq, optimizer_ntm, optimizer_whole = init_optimizers(model, ntm_model, opt)
        if opt.only_train_ntm or (opt.use_topic_represent and not opt.load_pretrain_ntm):
            train_ntm(ntm_model, optimizer_ntm, opt)
        else:
            print("Loading ntm model from %s" % opt.check_pt_ntm_model_path)
            if opt.train_from:
                model.load_state_dict(torch.load(opt.check_pt_seq_model_path))
            ntm_model.load_state_dict(torch.load(opt.check_pt_ntm_model_path))
            train_model(model, ntm_model, optimizer_seq2seq, optimizer_ntm, optimizer_whole, bow_dictionary, opt)
            training_time = time_since(start_time)
            logging.info('Time for training: %.1f' % training_time)
    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.my_own_opts(parser)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    logging = config.init_logging(log_file=opt.model_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
