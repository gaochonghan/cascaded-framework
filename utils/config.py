import os
import argparse
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class MInformerConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        # distributed settings
        parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend (default nccl)')
        parser.add_argument('--infi_band', type=str2bool, default=False, help='use infiniband')
        parser.add_argument('--infi_band_interface', default=0, type=int, help='default infiniband interface id')
        parser.add_argument('--world_size', type=int, default=-1, help='# of computation node')
        parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        parser.add_argument('--dist_file', default=None, type=str, help='url used to set up distributed training')
        parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
        parser.add_argument('--mp_dist', type=str2bool, default=True, help='allow multiple GPU on 1 node')
        parser.add_argument('--gpu', default=None, type=int, nargs='+', help='local GPU id to use')

        # informer settings
        parser.add_argument('--model', type=str, default='informer',
                            help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
        parser.add_argument('--data', type=str, default='ETTh1', help='data')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
        parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
        # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
        parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
        parser.add_argument('--cols', type=str, nargs='+', help='file list')
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=10, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training',
                            default=False)
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        parser.add_argument('--lambda_par', type=float, default=0.6)

        # other settings
        parser.add_argument('--w_momentum', type=float, default=0.9)
        parser.add_argument('--w_weight_decay', type=float, default=0.005)
        parser.add_argument('--A_lr', type=float, default=0.0002)
        parser.add_argument('--A_weight_decay', type=float, default=0)
        parser.add_argument('--max_hessian_grad_norm', type=float, default=1)
        parser.add_argument('--ratio', type=float, default=0.5)
        parser.add_argument('--sigmoid', type=float, default=1)
        parser.add_argument('--fourrier', action='store_true')
        parser.add_argument('--fourier_divider', type=int, default=100)
        parser.add_argument('--temp', type=float, default=5)
        parser.add_argument('--unrolled', type=float, default=1e-4)
        parser.add_argument('--trigger', action='store_true', default=False)

        # ablation study
        parser.add_argument('--teacher_head', type=int, default=8)
        parser.add_argument('--student_head', type=int, default=8)
        parser.add_argument('--noise', type=float, default=0.)

        # autoformer
        parser.add_argument('--moving_avg', type=int, default=15)
        parser.add_argument('--series_decomp', action='store_true')
        # SCINet
        parser.add_argument('--concat_len', type=int, default=0)
        parser.add_argument('--single_step', type=int, default=0)
        parser.add_argument('--single_step_output_One', type=int, default=0)
        parser.add_argument('--lastWeight', type=float, default=1.0)
        parser.add_argument('--hidden-size', default=1, type=float, help='hidden channel of module')
        parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
        parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
        parser.add_argument('--dilation', default=1, type=int, help='dilation')
        parser.add_argument('--window_size', default=12, type=int, help='input size')
        parser.add_argument('--positionalEcoding', type=bool, default=False)
        parser.add_argument('--groups', type=int, default=1)
        parser.add_argument('--levels', type=int, default=3)
        parser.add_argument('--stacks', type=int, default=1, help='1 stack or 2 stacks')
        parser.add_argument('--RIN', type=bool, default=False)

        # Query Selector
        parser.add_argument('--embedding_size', type=int, default=32)
        parser.add_argument('--setting', type=str, default='m_h1_48')
        parser.add_argument('--hidden_size', type=int, default=96)
        parser.add_argument('--encoder_attention', type=str, default="query_selector_0.90")
        parser.add_argument('--decoder_attention', type=str, default='full')

        args = parser.parse_args()

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.path = os.path.join("run", 'searchs')
        self._mk_folder(self.path)
        self.path = os.path.join(self.path, self.name)
        self._mk_folder(self.path)
        self.path = os.path.join(self.path, os.environ["SLURM_JOBID"])
        try:
            self._mk_folder(self.path)
        except FileExistsError:
            pass
        self.dist_path = os.path.join(self.path, 'dist')
        try:
            self._mk_folder(self.dist_path)
        except FileExistsError:
            pass
        # self.gpus = parse_gpus(self.gpus)

    def _mk_folder(self, path_in):
        path = os.path.abspath(path_in)
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
