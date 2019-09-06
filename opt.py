import argparse
'''
file to change the options

'''
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',type=int,default=25,help='number of workers in dataloader')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='minibatch size')
    # Data input settings
    parser.add_argument('--train_json',type=str,default='data/train_val_videodatainfo.json',
                        help='path to the json file containing train video info')

    parser.add_argument('--info_json',type=str,default='data/info.json',
                        help='path to the json file containing additional info and vocab')

    parser.add_argument( '--caption_json',type=str,default='data/caption.json',
                        help='path to the processed video caption json')

    parser.add_argument('--c3d_feats_dir', type=str, default='/home/sanjay/Documents/Video_convcap/train_val_feat.pkl')
    # parser.add_argument(
    #     '--with_c3d', type=int, default=0, help='whether to use c3d features')
    parser.add_argument( '--cached_tokens',type=str,default='msr-all-idxs',
                        help='Cached token file for calculating cider score \
                        during self critical training.')
    # Model settings
    parser.add_argument("--model", type=str, default='withoutattn', help="which model to use")

    parser.add_argument('--dim_vid',type=int, default=4096,
                        help='dim of features of video frames')
    
    parser.add_argument('--rnn_type', type=str, default='lstm', help='lstm or gru')
    
    parser.add_argument("--bidirectional",type=int,default=0,
                        help="0 for disable, 1 for enable. encoder/decoder bidirectional.")

    parser.add_argument('--dim_hidden',type=int, default=512,
                        help='size of the rnn hidden layer')
    
    parser.add_argument('--rnn_dropout_p',type=float,default=0.2,
                        help='strength of dropout in the Language Model RNN')

    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')

    parser.add_argument("--max_len",type=int,default=15,
                        help='max length of captions(containing <sos>,<eos>)')
    
  
    parser.add_argument('--input_dropout_p',type=float,default=0.2,
                        help='strength of dropout in the Language Model RNN')

    parser.add_argument('--dim_word',type=int,default=300,
                        help='the encoding size of each token in the vocabulary, and the video.')

   

    # Optimization: General

    
    
    parser.add_argument('--grad_clip',type=float,default=5,  # 5.,
                        help='clip gradients at this value')

    parser.add_argument('--self_crit_after',type=int,default=-1,
                        help='After what epoch do we start finetuning the CNN? \
                        (-1 = disable; never finetune, 0 = finetune from start)'
    )

    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')

    parser.add_argument('--learning_rate_decay_every',type=int,default=200,
                    help='every how many iterations thereafter to drop LR?(in epoch)')

    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)

    # parser.add_argument('--optim_alpha', type=float, default=0.9, help='alpha for adam')
    # parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    # parser.add_argument('--optim_epsilon',type=float,default=1e-8,
    #                         help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay',type=float,default=5e-4,
                        help='weight_decay. strength of weight regularization')

    parser.add_argument('--save_checkpoint_every',type=int,default=50,
                        help='how often to save a model checkpoint (in epoch)?')

    parser.add_argument('--checkpoint_path',type=str,default='checkpoint/',
                        help='directory to store checkpointed models')

    parser.add_argument('--gpu', type=str, default='1', help='gpu device number')

    args = parser.parse_args()
    print(args)

    return args
