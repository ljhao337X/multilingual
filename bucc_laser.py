import argparse
import os
import re
import logging
import sys
import tempfile
import time
from subprocess import run, DEVNULL
from typing import Optional

import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn

from fairseq.models.transformer import (
    Embedding,
    TransformerEncoder,
)
from fairseq.data.dictionary import Dictionary
from fairseq.modules import LayerNorm

SPACE_NORMALIZER = re.compile(r"\s+")
Batch = namedtuple("Batch", "srcs tokens lengths")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("preprocess")

_bpe_fcodes = './model/laser/93langs.fcodes'
_bpe_vocab = './model/laser/93langs.fvocab'

class LaserLstmEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings,
        padding_idx,
        embed_dim=320,
        hidden_size=512,
        num_layers=1,
        bidirectional=False,
        left_pad=True,
        padding_value=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, embed_dim, padding_idx=self.padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value
        )
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.output_units
                        )
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            "sentemb": sentemb,
            "encoder_out": (x, final_hiddens, final_cells),
            "encoder_padding_mask": encoder_padding_mask if encoder_padding_mask.any() else None,
        }


class SentenceEncoder:
    def __init__(
        self,
        model_path,
        max_sentences=None,
        max_tokens=None,
        cpu=False,
        fp16=False,
        verbose=False,
        sort_kind="quicksort",
    ):
        if verbose:
            logger.info(f"loading encoder: {model_path}")
        self.use_cuda = torch.cuda.is_available() and not cpu
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = torch.load(model_path)
        if "params" in state_dict:
            self.encoder = LaserLstmEncoder(**state_dict["params"])
            self.encoder.load_state_dict(state_dict["model"])
            self.dictionary = state_dict["dictionary"]
            self.prepend_bos = False
            self.left_padding = False

        del state_dict
        self.bos_index = self.dictionary["<s>"] = 0
        self.pad_index = self.dictionary["<pad>"] = 1
        self.eos_index = self.dictionary["</s>"] = 2
        self.unk_index = self.dictionary["<unk>"] = 3

        if fp16:
            self.encoder.half()
        if self.use_cuda:
            if verbose:
                logger.info("transfer encoder to GPU")
            self.encoder.cuda()
        self.encoder.eval()
        self.sort_kind = sort_kind

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        with torch.no_grad():
            sentemb = self.encoder(tokens, lengths)["sentemb"]
        embeddings = sentemb.detach().cpu().numpy()
        return embeddings

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ntokens = len(tokens)
        # 构建带<\s>的句子
        if self.prepend_bos:
            ids = torch.LongTensor(ntokens + 2)
            ids[0] = self.bos_index
            for i, token in enumerate(tokens):
                ids[i + 1] = self.dictionary.get(token, self.unk_index)
            ids[ntokens + 1] = self.eos_index
        else:
            ids = torch.LongTensor(ntokens + 1)
            for i, token in enumerate(tokens):
                ids[i] = self.dictionary.get(token, self.unk_index)
            ids[ntokens] = self.eos_index
        return ids

    def _make_batches(self, lines):
        # 每一句话中每个token的id
        tokens = [self._tokenize(line) for line in lines]
        # 每句话的长度, 包括了特殊符号
        lengths = np.array([t.numel() for t in tokens])
        # 给句子重新排序， 默认升序，因此得到的是降序的下标
        indices = np.argsort(-lengths, kind=self.sort_kind)

        def batch(tokens, lengths, indices):
            # 固定一个batch的长度
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]), self.pad_index)
            if not self.left_padding:
                for i in range(len(tokens)):
                    toks[i, : tokens[i].shape[0]] = tokens[i]
            else:
                for i in range(len(tokens)):
                    toks[i, -tokens[i].shape[0] :] = tokens[i]
            return (
                Batch(srcs=None, tokens=toks, lengths=torch.LongTensor(lengths)),
                indices,
            )

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and (
                (self.max_tokens is not None and ntokens + lengths[i] > self.max_tokens)
                or (self.max_sentences is not None and nsentences == self.max_sentences)
            ):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences):
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)
            results.append(self._process_batch(batch))

        # batch是从小到大排列的，再给它还原回去。
        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]


def load_model(
    encoder: str,
    verbose=False,
    **encoder_kwargs
) -> SentenceEncoder:
    spm_vocab = None
    return SentenceEncoder(
        encoder, verbose=verbose, **encoder_kwargs
    )


def BPEfastApply(inp_fname, out_fname,
                 verbose=False):
    if not os.path.isfile(out_fname):
        if verbose:
            logger.info('fastBPE: processing {}'.format(os.path.basename(inp_fname)))
        run(f"./tools-external/fastBPE/fast applybpe {out_fname} {inp_fname} {_bpe_fcodes} {_bpe_vocab}", shell=True, stderr=DEVNULL)


def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def EncodeTime(t):
    t = int(time.time() - t)
    if t < 1000:
        return "{:d}s".format(t)
    else:
        return "{:d}m{:d}s".format(t // 60, t % 60)
def EncodeFilep(
    encoder, inp_file, out_file, buffer_size=10000, fp16=False, verbose=False
):
    n = 0
    t = time.time()
    for sentences in buffered_read(inp_file, buffer_size):
        encoded = encoder.encode_sentences(sentences)
        if fp16:
            encoded = encoded.astype(np.float16)
        encoded.tofile(out_file)
        n += len(sentences)
        if verbose and n % 10000 == 0:
            logger.info("encoded {:d} sentences".format(n))
    if verbose:
        logger.info(f"encoded {n} sentences in {EncodeTime(t)}")


def EncodeFile(
    encoder,
    inp_fname,
    out_fname,
    buffer_size=10000,
    fp16=False,
    verbose=False,
    over_write=False,
    inp_encoding="utf-8",
):
    # TODO :handle over write
    if not os.path.isfile(out_fname):
        if verbose:
            logger.info(
                "encoding {} to {}".format(
                    inp_fname if len(inp_fname) > 0 else "stdin", out_fname,
                )
            )
        fin = (
            open(inp_fname, "r", encoding=inp_encoding, errors="surrogateescape")
            if len(inp_fname) > 0
            else sys.stdin
        )
        fout = open(out_fname, mode="wb")
        EncodeFilep(
            encoder, fin, fout, buffer_size=buffer_size, fp16=fp16, verbose=verbose
        )
        fin.close()
        fout.close()
    elif not over_write and verbose:
        logger.info("encoder: {} exists already".format(os.path.basename(out_fname)))


def embed_sentences(
    encoder_path: str = None,
    languages = ['zh'],
    data_type: str = 'sample',
    verbose: bool = False,
    max_tokens: int = 12000,
    max_sentences: Optional[int] = None,
    cpu: bool = False,
    sort_kind: str = "quicksort",
):
    encoder = load_model(
        encoder_path,
        verbose=verbose,
        max_sentences=max_sentences,
        max_tokens=max_tokens,
        sort_kind=sort_kind,
        cpu=cpu,
    )

    for language in languages:
        if data_type == 'sample':
            input_folder = f'./dataset/bucc/bucc2018/{language}-en/'
        else:
            input_folder = f'./dataset/bucc/bucc2017/{language}-en/'

        for x in [language, 'en']:
            if data_type == 'sample':
                input_raw_file = input_folder + f'{language}-en.sample.{x}'
                output = input_folder + f'laser_{x}_sample_5.npy'
            else:
                input_raw_file = input_folder + f'{language}-en.training.{x}'
                output = input_folder + f'laser_{x}_5.npy'

            with tempfile.TemporaryDirectory() as tmpdir:
                bpe_fname = os.path.join(tmpdir, "bpe")
                # process input
                sents = []
                with open(input_raw_file, 'r', encoding='utf-8') as f:
                    while True:
                        line = f.readline()
                        if line:
                            sents.append(line.split('\t')[1])
                        else:
                            break
                preprocessed_input = os.path.join(tmpdir, 'txt')
                with open(preprocessed_input, 'w', encoding='utf-8') as f:
                    f.writelines(sents)
                BPEfastApply(
                    preprocessed_input, bpe_fname, verbose=verbose
                )

                EncodeFile(
                    encoder,
                    bpe_fname,
                    output,
                    verbose=verbose,
                )
                torch.cuda.empty_cache()


if __name__ == '__main__':
    # languages = ['zh', 'ru', 'de', 'fr']
    languages = ['zh']
    data_type = 'training'
    encoder_path = './model/laser/bilstm.93langs.2018-12-26.pt'
    embed_sentences(
        languages=languages,
        data_type=data_type,
        encoder_path=encoder_path,
        verbose=True,
    )

    # tokenize
    # command = './tools-external/fastBPE/fast applybpe ./dataset/bucc/bucc2018/zh-en/sample_test.txt ./dataset/bucc/bucc2018/zh-en/zh-en.sample.zh ./model/laser/93langs.fcodes ./model/laser/93langs.fvocab'