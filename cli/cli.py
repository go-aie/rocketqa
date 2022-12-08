# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil

import paddle
import rocketqa


def train(base_model, train_set, use_cuda, epoch, out, **kwargs):
    # Train the model and save the result into a temp directory
    temp_path = os.path.join(out, 'temp')
    encoder = rocketqa.load_model(model=base_model, use_cuda=use_cuda,
                                  device_id=0, batch_size=32)
    encoder.train(train_set, epoch, temp_path, **kwargs)

    # Remove unused moment files
    model_checkpoint_path = glob.glob(os.path.join(temp_path, 'step_*'))[0]
    moment_filenames = os.path.join(model_checkpoint_path, '*moment*')
    for f in glob.glob(moment_filenames):
        os.remove(f)

    # Move checkpoint files into a new directory
    dest_model_checkpoint_path = os.path.join(out, get_model_checkpoint_path(
        base_model))
    shutil.move(model_checkpoint_path, dest_model_checkpoint_path)

    # Remove the empty temp directory
    os.rmdir(temp_path)

    # Create or update the model config files
    def create_if_not_exists(path, filename, gen_content):
        fullpath = os.path.join(path, filename)
        if not os.path.exists(fullpath):
            with open(fullpath, 'w') as f:
                f.write(gen_content())

    create_if_not_exists(out, 'config.json',
                         lambda: get_model_config(base_model))
    create_if_not_exists(out, 'zh_config.json', lambda: get_model_zh_config())
    create_if_not_exists(out, 'zh_vocab.txt',
                         lambda: get_model_zh_vocab(base_model))


def get_model_checkpoint_path(model):
    if model.endswith(('_de', '_de_v2')):
        return 'dual_params'
    elif model.endswith(('_ce', '_ce_v2')):
        return 'cross_params'
    else:
        return ''


def get_model_config(model):
    if model.endswith(('_de', '_de_v2')):
        # Dual Encoder
        return '''{
  "model_type": "dual_encoder",
  "q_max_seq_len": 32,
  "p_max_seq_len": 384,
  "model_conf_path": "zh_config.json",
  "model_vocab_path": "zh_vocab.txt",
  "model_checkpoint_path": "dual_params",
  "for_cn": true,
  "share_parameter": 0
}'''
    elif model.endswith(('_ce', '_ce_v2')):
        # Cross Encoder
        return '''{
  "model_type": "cross_encoder",
  "max_seq_len": 384,
  "model_conf_path": "zh_config.json",
  "model_vocab_path": "zh_vocab.txt",
  "model_checkpoint_path": "cross_params",
  "for_cn": true,
  "share_parameter": 0
}'''
    else:
        return ''


def get_model_zh_config():
    return '''{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "relu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "max_position_embeddings": 513,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 18000
}'''


def get_model_zh_vocab(model):
    vocab_filename = os.path.join(os.path.expanduser('~/.rocketqa/'), model,
                                  'zh_vocab.txt')
    if not os.path.exists(vocab_filename):
        return ""
    with open(vocab_filename) as f:
        return f.read()


def save(model, out):
    encoder = rocketqa.load_model(
        model=model,
        use_cuda=False,
        device_id=0,
        batch_size=32,
    )

    if isinstance(encoder, rocketqa.DualEncoder):
        save_de_model(encoder, out)
    elif isinstance(encoder, rocketqa.CrossEncoder):
        save_ce_model(encoder, out)


def save_de_model(dual_encoder, out):
    feeded_var_names = [
        'read_file_0.tmp_0',
        'read_file_0.tmp_1',
        'read_file_0.tmp_2',
        # 'read_file_0.tmp_3',
        'read_file_0.tmp_4',
        'read_file_0.tmp_5',
        'read_file_0.tmp_6',
        'read_file_0.tmp_7',
        # 'read_file_0.tmp_8',
        'read_file_0.tmp_9',
        # 'read_file_0.tmp_10',
        # 'read_file_0.tmp_11',
    ]
    block = dual_encoder.test_prog.global_block()
    feed_vars = [block.var(name) for name in feeded_var_names]

    target_vars = [dual_encoder.graph_vars["q_rep"],
                   dual_encoder.graph_vars["p_rep"]]

    paddle.static.save_inference_model(
        path_prefix=out or dual_encoder.args.model_name,
        feed_vars=feed_vars,
        fetch_vars=target_vars,
        executor=dual_encoder.exe,
        program=dual_encoder.test_prog,
    )


def save_ce_model(cross_encoder, out):
    feeded_var_names = [
        "read_file_0.tmp_0",
        "read_file_0.tmp_1",
        "read_file_0.tmp_2",
        # "read_file_0.tmp_3",
        "read_file_0.tmp_4",
        # "read_file_0.tmp_5",
        # "read_file_0.tmp_6",
    ]
    block = cross_encoder.test_prog.global_block()
    feed_vars = [block.var(name) for name in feeded_var_names]

    target_vars = [cross_encoder.graph_vars["probs"]]

    paddle.static.save_inference_model(
        path_prefix=out or cross_encoder.args.model_name,
        feed_vars=feed_vars,
        fetch_vars=target_vars,
        executor=cross_encoder.exe,
        program=cross_encoder.test_prog,
    )


def main():
    parser = argparse.ArgumentParser(prog='rocketqa')
    subparsers = parser.add_subparsers(dest='command', title='commands')

    train_parser = subparsers.add_parser('train',
                                         help='train or finetune the dual/cross encoder model')
    train_parser.add_argument('base_model',
                              choices=rocketqa.available_models(),
                              help='base model', metavar='base_model')
    train_parser.add_argument('train_set', help='train set')
    train_parser.add_argument('--use-cuda', action='store_true',
                              help='whether to run models on GPU (default: %(default)s)')
    train_parser.add_argument('--epoch', type=int, default=2,
                              help='epoch (default: %(default)s)')
    train_parser.add_argument('--out-path', default='./models',
                              help='output directory (default: %(default)s)')
    train_parser.add_argument('--save-steps', type=int, default=1000,
                              help='save steps (default: %(default)s)')
    train_parser.add_argument('--learning-rate', type=float, default=1e-5,
                              help='learning rate (default: %(default)s)')

    save_parser = subparsers.add_parser('save',
                                        help='save the inference model from raw dual/cross encoder model')
    save_parser.add_argument('model', help='model config')
    save_parser.add_argument('--out-path',
                              help='output path without suffix (default: <input_model_name>')

    args = parser.parse_args()
    if args.command == 'train':
        train(args.base_model, args.train_set, args.use_cuda, args.epoch,
              args.out_path,
              save_steps=args.save_steps, learning_rate=args.learning_rate)
    elif args.command == 'save':
        save(args.model, args.out_path)


if __name__ == '__main__':
    main()