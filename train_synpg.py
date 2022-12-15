import os, argparse, h5py, codecs
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk import ParentedTree
from subwordnmt.apply_bpe import BPE, read_vocabulary
from model import SynPG
from utils import Timer, make_path, load_data, load_embedding, load_dictionary, deleaf, sent2str, synt2str
from pprint import pprint, pformat

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="./model/", 
                       help="directory to save models")
parser.add_argument('--load_model', type=str, default="none",
                       help="load pretrained model")
parser.add_argument('--output_dir', type=str, default="auto",
                       help="directory to save outputs")
parser.add_argument('--bpe_codes_path', type=str, default='./data/bpe.codes',
                       help="bpe codes file")
parser.add_argument('--bpe_vocab_path', type=str, default='./data/vocab.txt',
                       help="bpe vcocabulary file")
parser.add_argument('--bpe_vocab_thresh', type=int, default=50, 
                       help="bpe threshold")
parser.add_argument('--dictionary_path', type=str, default="./data/dictionary.pkl", 
                       help="dictionary file")
parser.add_argument('--train_data_path', type=str, default="./data/train_data.h5",
                       help="training data")
parser.add_argument('--train_data_subset', type=int, default=-1,
                       help="training data subset")
parser.add_argument('--valid_data_path', type=str, default="./data/valid_data.h5",
                       help="validation data")
parser.add_argument('--emb_path', type=str, default="./data/glove.840B.300d.txt", 
                       help="initialized word embedding")
parser.add_argument('--max_sent_len', type=int, default=40,
                       help="max length of sentences")
parser.add_argument('--max_synt_len', type=int, default=160,
                       help="max length of syntax")
parser.add_argument('--word_dropout', type=float, default=0.4,
                       help="word dropout ratio")
parser.add_argument('--n_epoch', type=int, default=5,
                       help="number of epoches")
parser.add_argument('--batch_size', type=int, default=64,
                       help="batch size")
parser.add_argument('--lr', type=float, default=1e-4,
                       help="learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help="weight decay for adam")
parser.add_argument('--log_interval', type=int, default=250,
                       help="print log and validation loss evry 250 iterations")
parser.add_argument('--gen_interval', type=int, default=5000,
                       help="generate outputs every 500 iterations")
parser.add_argument('--save_interval', type=int, default=10000,
                       help="save model every 10000 iterations")
parser.add_argument('--temp', type=float, default=0.5,
                       help="temperature for generating outputs")
parser.add_argument('--seed', type=int, default=0, 
                       help="random seed")
parser.add_argument('--gpuid', type=str, default="-1",
                       help="cuda visible devices")
args = parser.parse_args()

if args.output_dir.lower() == "auto":
    args.output_dir = os.path.join(args.model_dir, "output")

# create folders
make_path(args.model_dir)
make_path(args.output_dir)

if args.gpuid != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

import logging, time
logging.basicConfig(
    handlers=[
        logging.FileHandler(
            filename="./{}/log-train-{}.log".format(
                args.model_dir,
                time.strftime("%Y-%m-%dT%H%M%S", time.gmtime())
            ),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ],
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

logger.info(pformat(vars(args)))

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

def train(epoch, model, train_data, valid_data, train_loader, valid_loader, optimizer, criterion, dictionary, bpe, args):
    
    timer = Timer()
    n_it = len(train_loader)
    num_sent_too_long = 0
    
    for it, data_idxs in enumerate(train_loader):
        model.train()
        
        data_idxs = np.sort(data_idxs.numpy())
        
        # get batch of raw sentences and raw syntax
        # sents_ = train_data[0][data_idxs]
        # synts_ = train_data[1][data_idxs]
        sents_ = [train_data[0][i_] for i_ in data_idxs]
        synts_ = [train_data[1][i_] for i_ in data_idxs]

        # 不同数据，HDF5 文件读出来的格式不一样。LLJ
        # if not isinstance(sents_[0], str):
        #     sents_ = [s.decode("utf-8") for s in sents_]
        #     synts_ = [s.decode("utf-8") for s in synts_]
            
        batch_size = len(sents_)
        
        # initialize tensors
        sents = np.zeros((batch_size, args.max_sent_len), dtype=np.long)    # words without position
        synts = np.zeros((batch_size, args.max_synt_len+2), dtype=np.long)  # syntax
        targs = np.zeros((batch_size, args.max_sent_len+2), dtype=np.long)  # target output
        
        for i in range(batch_size):
            if len(sents_[i]) > sents.shape[1] or len(synts_[i]) > synts.shape[1]:
                num_sent_too_long += 1
                continue
            # bpe segment and convert to tensor
            sent_ = sents_[i]
            # sent_ = bpe.segment(sent_).split()
            # sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
            sents[i, :len(sent_)] = sent_
            
            # add <sos> and <eos> for target output
            targ_ = [dictionary.word2idx["<sos>"]] + sent_ + [dictionary.word2idx["<eos>"]]
            targs[i, :len(targ_)] = targ_
            
            # parse syntax and convert to tensor
            synt_ = synts_[i]
            # synt_ = ParentedTree.fromstring(synt_)
            # synt_ = deleaf(synt_)
            # synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
            # synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
            synts[i, :len(synt_)] = synt_
            
        sents = torch.from_numpy(sents).cuda()
        synts = torch.from_numpy(synts).cuda()
        targs = torch.from_numpy(targs).cuda()
        
        # forward
        outputs = model(sents, synts, targs)
        
        # calculate loss
        targs_ = targs[:, 1:].contiguous().view(-1)
        outputs_ = outputs.contiguous().view(-1, outputs.size(-1))
        optimizer.zero_grad()
        loss = criterion(outputs_, targs_)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if it % args.log_interval == 0:
            # print current loss
            valid_loss = evaluate(model, valid_data, valid_loader, criterion, dictionary, bpe, args)
            logger.info("| ep {:2d}/{} | it {:3d}/{} | {:5.2f} s | loss {:.4f} | g_norm {:.6f} | valid loss {:.4f} |".format(
                epoch, args.n_epoch, it, n_it, timer.get_time_from_last(), loss.item(), model.grad_norm, valid_loss))
            
        if it % args.gen_interval == 0:
            # generate output to args.output_dir
            generate(epoch, it, model, valid_data, valid_loader, dictionary, bpe, args)
            
        if it % args.save_interval == 0:
            # save model to args.model_dir
            torch.save(model.state_dict(), os.path.join(args.model_dir, "synpg_epoch{:02d}.pt".format(epoch)))
    # logger.info("num_sent_too_long: ", num_sent_too_long)
            
def evaluate(model, data, loader, criterion, dictionary, bpe, args):
    model.eval()
    total_loss = 0.0
    max_it = len(loader)
    with torch.no_grad():
        for it, data_idxs in enumerate(loader):
            data_idxs = np.sort(data_idxs.numpy())
            
            # get batch of raw sentences and raw syntax
            # sents_ = data[0][data_idxs]
            # synts_ = data[1][data_idxs]
            sents_ = [data[0][i_] for i_ in data_idxs]
            synts_ = [data[1][i_] for i_ in data_idxs]

            batch_size = len(sents_)
            
            # initialize tensors
            sents = np.zeros((batch_size, args.max_sent_len), dtype=np.long)    # words without position
            synts = np.zeros((batch_size, args.max_synt_len+2), dtype=np.long)  # syntax
            targs = np.zeros((batch_size, args.max_sent_len+2), dtype=np.long)  # target output

            for i in range(batch_size):
                
                # bpe segment and convert to tensor
                sent_ = sents_[i]
                # sent_ = bpe.segment(sent_).split()
                # sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
                sents[i, :len(sent_)] = sent_
                
                # add <sos> and <eos> for target output
                targ_ = [dictionary.word2idx["<sos>"]] + sent_ + [dictionary.word2idx["<eos>"]]
                targs[i, :len(targ_)] = targ_
                
                # parse syntax and convert to tensor
                synt_ = synts_[i]
                # synt_ = ParentedTree.fromstring(synt_)
                # synt_ = deleaf(synt_)
                # synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
                # synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
                synts[i, :len(synt_)] = synt_

            sents = torch.from_numpy(sents).cuda()
            synts = torch.from_numpy(synts).cuda()
            targs = torch.from_numpy(targs).cuda()
            
            # forward
            outputs = model(sents, synts, targs)
            
            # calculate loss
            targs_ = targs[:, 1:].contiguous().view(-1)
            outputs_ = outputs.contiguous().view(-1, outputs.size(-1))
            loss = criterion(outputs_, targs_)
        
            total_loss += loss.item()
    
    return total_loss / max_it

def generate(epoch, eit, model, data, loader, dictionary, bpe, args, max_it=10):
    model.eval()
    with open(os.path.join(args.output_dir, "sents_valid_epoch{:02d}_it{:06d}.txt".format(epoch, eit)), "w") as fp:
        with torch.no_grad():
            for it, data_idxs in enumerate(loader):
                if it >= max_it:
                    break
                
                data_idxs = np.sort(data_idxs.numpy())
                
                # get batch of raw sentences and raw syntax
                # sents_ = data[0][data_idxs]
                # synts_ = data[1][data_idxs]
                sents_ = [data[0][i_] for i_ in data_idxs]
                synts_ = [data[1][i_] for i_ in data_idxs]

                batch_size = len(sents_)
                
                # initialize tensors
                sents = np.zeros((batch_size, args.max_sent_len), dtype=np.long)    # words without position
                synts = np.zeros((batch_size, args.max_synt_len+2), dtype=np.long)  # syntax
                targs = np.zeros((batch_size, args.max_sent_len+2), dtype=np.long)  # target output

                for i in range(batch_size):
                    
                    # bpe segment and convert to tensor
                    sent_ = sents_[i]
                    # sent_ = bpe.segment(sent_).split()
                    # sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
                    sents[i, :len(sent_)] = sent_
                    
                    # add <sos> and <eos> for target output
                    targ_ = [dictionary.word2idx["<sos>"]] + sent_ + [dictionary.word2idx["<eos>"]]
                    targs[i, :len(targ_)] = targ_
                    
                    # parse syntax and convert to tensor
                    synt_ = synts_[i]
                    # synt_ = ParentedTree.fromstring(synt_)
                    # synt_ = deleaf(synt_)
                    # synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
                    # synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
                    synts[i, :len(synt_)] = synt_
            
                sents = torch.from_numpy(sents).cuda()
                synts = torch.from_numpy(synts).cuda()
                targs = torch.from_numpy(targs).cuda()
                
                # generate
                idxs = model.generate(sents, synts, sents.size(1), temp=args.temp)
                
                # write output
                for sent, idx, synt in zip(sents.cpu().numpy(), idxs.cpu().numpy(), synts.cpu().numpy()):
                    fp.write(synt2str(synt[1:], dictionary)+'\n')
                    fp.write(sent2str(sent, dictionary)+'\n')
                    fp.write(synt2str(idx, dictionary)+'\n')
                    fp.write("--\n")

logger.info("==== loading data ====")

# load bpe codes
bpe_codes = codecs.open(args.bpe_codes_path, encoding='utf-8')
bpe_vocab = codecs.open(args.bpe_vocab_path, encoding='utf-8')
bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

# load dictionary and data
dictionary = load_dictionary(args.dictionary_path)

def preprocess_to_pt(file: str, args):
    """
    把 train 函数里的一些公共步骤提取出来。
    提前进行 BPE 分词、parse format、token 映射 id 等操作。
    不用每次训练的时候都重新做一次。
    并且把预处理数据用 torch.save 保存，以后直接读取，无需再次处理。
    by LLJ.
    """

    if os.path.exists(file + ".preprocessed.pt"):
        logger.info("File has been preprocessed.")
        return

    logger.info("Loading data")
    if file.endswith(".h5"):
        data = load_data(file)
    elif file.endswith(".txt"):
        with open(file, "r") as f:
            data = [s.strip().split("\t") for s in f.readlines()]
            data = list(zip(*data))

    logger.info("Processing data")
    sents_, synts_ = data[0], data[1]

    # 不同数据，HDF5 文件读出来的格式不一样。格式转换的时候，h5py 的版本 v2 和 v3 又不同，好麻烦。LLJ
    if not isinstance(sents_[0], str):
        logger.info("Decoding string")
        try:
            sents_ = sents_.asstr()
            synts_ = synts_.asstr()
        except Exception as e:
            logger.info(e)
            sents_ = [s.decode("utf-8") for s in tqdm(sents_)]
            synts_ = [s.decode("utf-8") for s in tqdm(synts_)]

    sents_fit_max_len, synts_fit_max_len = [], []
    logger.info("BPE, tokenize, map to id")
    for i in tqdm(range(len(sents_)), total=len(sents_)):
        try:
            # sents
            sent_ = sents_[i]
            sent_ = bpe.segment(sent_).split()
            sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
            # synts
            synt_ = synts_[i]
            synt_ = ParentedTree.fromstring(synt_)
            synt_ = deleaf(synt_)
            synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
            synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
            sents_fit_max_len.append(sent_)
            synts_fit_max_len.append(synt_)
        except Exception as e:
            # 有些数据是乱码，会出错，直接填充空的数据。不能丢弃，因为数据行号和 id 有关联。
            sents_fit_max_len.append([dictionary.word2idx["<sos>"]] + [dictionary.word2idx["<eos>"]])
            synts_fit_max_len.append([dictionary.word2idx["<sos>"]] + [dictionary.word2idx["<eos>"]])
            pass

    logger.info("{}/{} sentences of length within max length.".format(len(sents_fit_max_len), len(sents_)))

    temp = {"sents": sents_fit_max_len, "synts": synts_fit_max_len}
    torch.save(temp, file + ".preprocessed.pt")

    return


train_data_preprocessed, valid_data_preprocessed = False, False

if not os.path.exists(args.train_data_path + ".preprocessed.pt"):
    preprocess_to_pt(args.train_data_path, args)

if not os.path.exists(args.valid_data_path + ".preprocessed.pt"):
    preprocess_to_pt(args.valid_data_path, args)

train_data = torch.load(args.train_data_path + ".preprocessed.pt")
train_data = (train_data["sents"], train_data["synts"])

valid_data = torch.load(args.valid_data_path + ".preprocessed.pt")
valid_data = (valid_data["sents"], valid_data["synts"])

if args.train_data_subset != -1:
    train_data = (train_data[0][:args.train_data_subset], train_data[1][:args.train_data_subset])

train_idxs = np.arange(len(train_data[0]))
valid_idxs = np.arange(len(valid_data[0]))
logger.info(f"number of train examples: {len(train_idxs):,}")
logger.info(f"number of valid examples: {len(valid_idxs):,}")

train_loader = DataLoader(train_idxs, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_idxs, batch_size=args.batch_size, shuffle=False)

# build model and load initialized glove embedding
# Bug to fix: words that are not in the pretrained glove embedding will be all zero initialized. (Linjian Li)
embedding = load_embedding(args.emb_path, dictionary)
model = SynPG(len(dictionary), 300, word_dropout=args.word_dropout)
model.load_embedding(embedding)

if args.load_model.lower() != "none":
    logger.info("load pretrained model: {}".format(args.load_model))
    model.load_state_dict(torch.load(args.load_model))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss(ignore_index=dictionary.word2idx["<pad>"])

model = model.cuda()
criterion = criterion.cuda()

total_params = 0
for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue
    num_param = parameter.numel()
    total_params += num_param
logger.info("Total Trainable Params: {:,}".format(total_params))

logger.info("==== start training ====")
for epoch in range(1, args.n_epoch+1):
    # training
    start = time.time()
    train(epoch, model, train_data, valid_data, train_loader, valid_loader, optimizer, criterion, dictionary, bpe, args)
    time_elapsed = time.time() - start
    logger.info("Time elapsed for epoch {}: {}".format(epoch, time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))
    # save model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "synpg_epoch{:02d}.pt".format(epoch)))
    logger.info("Model saved to: {}".format(os.path.join(args.model_dir, "synpg_epoch{:02d}.pt".format(epoch))))
    # shuffle training data
    train_loader = DataLoader(train_idxs, batch_size=args.batch_size, shuffle=True)
