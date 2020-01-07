import sys, json, os, argparse, logging, torch
import numpy as np
import opennre

def download_dataset(args):
    if args.dataset is not None:
        if args.dataset in ['wiki80', 'semeval']:
            opennre.download(args.dataset, root_path=args.root_path)
        elif args.dataset == 'tacred':
            logging.warning('TACRED is released via the Linguistic Data Consortium (LDC). Please download it from https://catalog.ldc.upenn.edu/LDC2018T24')
        else:
            raise Exception('For sentence-level RE, Dataset must be one of [`wiki80`, `tacred`, `semeval`].')

def download_pretrain(args):
    if 'bert' in args.encoder:
        opennre.download('bert_base_uncased', root_path=args.root_path)
    elif 'cnn' in args.encoder:
        opennre.download('glove', root_path=args.root_path)

def main():
    parser = argparse.ArgumentParser()
    
    # Assign dataset OR train, test, val, and rel2id file path
    parser.add_argument('--dataset', type=str, default=None,
            help='Dataset name. Choose from `wiki80`, `semeval`, `tacred`.')
    parser.add_argument('--train', type=str, default=None, help='Data file for training.')
    parser.add_argument('--val', type=str, default=None, help='Data file for validation.')
    parser.add_argument('--test', type=str, default=None, help='Data file for test.')
    parser.add_argument('--rel2id', type=str, default=None, help='Relation -> id mapping.')
    
    # Model
    parser.add_argument('--encoder', type=str, default='bert',
            help='Sentence encoder name. Choose from `bert`, `bertentity`, `cnn`.')
    parser.add_argument('--optim', type=str, default=None,
            help='Optimizer. Choose from `sgd`, `adam`, `adamw` (for bert).')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=None, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=None, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--max_length', type=int, default=128, help='Max length.')
    parser.add_argument('--max_epoch', type=int, default=None, help='Max epoch.')
    
    # Others
    parser.add_argument('--ckpt', default=None, help='Checkpoint name.')
    parser.add_argument('--metric', default=None, help='Metric for validation, `acc` or `micro_f1`.')
    parser.add_argument('--root_path', default='.', 
            help='Working directory where you store the datasets and pre-trained parameters.')
    parser.add_argument('--mask_entity', action='store_true', 
            help='Mask entity mentions with a special token.')
    
    args = parser.parse_args()
    
    # Fix seed
    opennre.fix_seed()

    # Path settings
    sys.path.append(args.root_path)
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    
    # Download 
    logging.info('Downloading datasets and pretrained parameters.')
    download_dataset(args)
    download_pretrain(args)
    
    # Set dataset path
    if args.dataset:
        rel2id = json.load(open(os.path.join(args.root_path, 'benchmark/{}/{}_rel2id.json'
            .format(args.dataset, args.dataset))))
        args.train = os.path.join(args.root_path, 'benchmark/{}/{}_train.txt'
                .format(args.dataset, args.dataset))
        args.val = os.path.join(args.root_path, 'benchmark/{}/{}_val.txt'
                .format(args.dataset, args.dataset))
        args.test = os.path.join(args.root_path, 'benchmark/{}/{}_test.txt'
                .format(args.dataset, args.dataset))
        if not os.path.exists(args.val):
            args.val = args.test
        if not os.path.exists(args.test):
            args.test = args.val
    else:
        rel2id = json.load(open(args.rel2id))
    
    # Set checkpoint save path
    if args.ckpt:
        args.ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)
    else:
        args.ckpt = 'ckpt/{}_{}_softmax.pth.tar'.format(args.dataset, args.encoder)
    
    # Define the sentence encoder
    logging.info('Initializing the sentence encoder.')
    if args.encoder == 'bert':
        sentence_encoder = opennre.encoder.BERTEncoder(
            max_length=args.max_length, 
            pretrain_path=os.path.join(args.root_path, 'pretrain/bert-base-uncased'),
            mask_entity=args.mask_entity
        )
        args.lr = args.lr or 2e-5
        args.optim = args.optim or 'adamw'
        args.max_epoch = args.max_epoch or 10
    elif args.encoder == 'bertentity':
        sentence_encoder = opennre.encoder.BERTEntityEncoder(
            max_length=args.max_length, 
            pretrain_path=os.path.join(args.root_path, 'pretrain/bert-base-uncased'),
            mask_entity=args.mask_entity
        )
        args.lr = args.lr or 2e-5
        args.optim = args.optim or 'adamw'
        args.max_epoch = args.max_epoch or 10
    elif args.encoder == 'cnn':
        word2id = json.load(open(os.path.join(args.root_path, 
            'pretrain/glove/glove.6B.50d_word2id.json')))
        word2vec = np.load(os.path.join(args.root_path, 
            'pretrain/glove/glove.6B.50d_mat.npy'))
        sentence_encoder = opennre.encoder.CNNEncoder(
            token2id=word2id, max_length=args.max_length, word_size=50, position_size=5,
            hidden_size=230, blank_padding=True, kernel_size=3, padding_size=1,
            word2vec=word2vec, dropout=0.5
        )
        args.lr = args.lr or 1e-1
        args.wd = args.wd or 1e-5
        args.optim = args.optim or 'sgd'
        args.max_epoch = args.max_epoch or 100
    else:
        raise Exception('Encoder must be one of [`bert`, `bertentity`, `cnn`].')
    
    # Define the model
    logging.info('Initializing the model.')
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    
    # Define the whole training framework
    logging.info('Initializing the training framework.')
    framework = opennre.framework.SentenceRE(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        model=model,
        ckpt=args.ckpt,
        batch_size=args.batch_size, 
        max_epoch=args.max_epoch,
        lr=args.lr,
        weight_decay=args.wd,
        opt=args.optim
    )

    # Metric
    if args.dataset in ['tacred', 'semeval']:
        args.metric = args.metric or 'micro_f1'
    else:
        args.metric = args.metric or 'acc'

    # Print hyperparameters
    logging.info('dataset: {}'.format(args.dataset))
    logging.info('encoder: {}'.format(args.encoder))
    logging.info('batch size: {}'.format(args.batch_size))
    logging.info('max epoch: {}'.format(args.max_epoch))
    logging.info('learning rate: {}'.format(args.lr))
    logging.info('weight decay: {}'.format(args.wd))
    logging.info('optimizer: {}'.format(args.optim))
    logging.info('checkpoint path: {}'.format(args.ckpt))
    logging.info('metric: {}'.format(args.metric))
    
    # Train the model
    logging.info('Start training.')
    framework.train_model(metric=args.metric)
    
    # Test the model
    logging.info('Start testing.')
    framework.load_state_dict(torch.load(args.ckpt)['state_dict'])
    result = framework.eval_model(framework.test_loader)
    if args.metric == 'acc':
        logging.info('Accuracy on test set: {}'.format(result['acc']))
    else:
        logging.info('Micro Precision on test set: {}'.format(result['micro_p']))
        logging.info('Micro Recall on test set: {}'.format(result['micro_r']))
        logging.info('Micro F1 on test set: {}'.format(result['micro_f1']))

if __name__ == '__main__':
    main()
