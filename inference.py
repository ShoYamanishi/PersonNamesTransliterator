#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import json

import unicodedata
import re
import os
import io
import sys
import time
import math
import argparse

from RedBlackTree.rbtree import RedBlackNode
from RedBlackTree.rbtree import RedBlackTree
from StackDecoder.stack_decoder import StackDecoderPath
from StackDecoder.stack_decoder import StackDecoder

import utils.character_converters as cc

class CommandParameterServer():

    def __init__(self):

        self.DEFAULT_MODEL         = 'training_output/alpha_to_kana_128_wo_attn/ckpt-105'
        self.DEFAULT_TOKENIZER_DIR = 'training_data'
        self.MAX_LENGTH_ALPHAS     = 18
        self.MAX_LENGTH_KANAS      = 14

        argparser                  = self.construct_argparser()
        parsed_args                = argparser.parse_args()

        self.model_path            = parsed_args.model_path_prefix
        self.fp_in                 = parsed_args.infile
        self.fp_out                = parsed_args.outfile
        self.tokenizer_dir         = parsed_args.tokenizer_dir

        if parsed_args.use_attention:
            self.use_attention = True

        elif parsed_args.do_not_use_attention:
            self.use_attention = False

        else:
            self.use_attention = ( re.match(r'.*wo_attn.*', self.model_path) == None )


        if parsed_args.alpha_to_kana:
            self.direction = 'alpha_to_kana'

        elif parsed_args.kana_to_alpha:
            self.direction = 'kana_to_alpha'

        elif re.match(r'.*alpha_to_kana.*', self.model_path):
            self.direction = 'alpha_to_kana'

        else:
            self.direction = 'kana_to_alpha'

        if parsed_args.num_units == -1:
            self.num_units = int ( re.search( r'.*(alpha_to_kana_|kana_to_alpha_)(\d+)(_|/|\\).*', self.model_path ).group(2) )

        self.beam_width            = parsed_args.beam_width
        self.nbest                 = parsed_args.nbest
        self.max_length_alphas     = self.MAX_LENGTH_ALPHAS
        self.max_length_kanas      = self.MAX_LENGTH_KANAS
        self.tokenizer_path_alphas = self.tokenizer_dir + '/alphas_tokenizer.json'
        self.tokenizer_path_kanas  = self.tokenizer_dir + '/kanas_tokenizer.json'


    def construct_argparser(self):

        arg_parser = argparse.ArgumentParser( 
            description = 'transliterates western person names in Latin-based alphabets into Japanese Kana' 
        )

        arg_parser.add_argument( '--infile', '-i', 
            nargs = '?', type = argparse.FileType( 'r', encoding='utf-8' ), default = sys.stdin,
            help = 'file that contains list of input names'
        )

        arg_parser.add_argument( '--outfile', '-o',
            nargs = '?', type = argparse.FileType( 'w', encoding='utf-8' ), default = sys.stdout,
            help = 'output file'
        )

        arg_parser.add_argument( '--model_path_prefix', '-m',
            nargs = '?', type =str, default = self.DEFAULT_MODEL, 
            help = 'path to the model data file without extension'
        )

        arg_parser.add_argument( '--tokenizer_dir', '-t', 
            nargs = '?', type =str, default = self.DEFAULT_TOKENIZER_DIR,
            help = 'directory that contains the tokenizer files in json. The names for the tokenizers are{alphas/kanas_tokenizer.json'
        )

        arg_parser.add_argument( '--use_attention', '-pa',
            action="store_true",
            help = 'set this flag if the model\'s decoder uses attention'
        )

        arg_parser.add_argument( '--do_not_use_attention', '-na',
            action="store_true",
            help = 'set this flag if the model\'s decoder does not use attention. If nether -pa or -na is set, inferred from the model path prefix.'
        )

        arg_parser.add_argument( '--num_units', '-u',
            type = int, nargs = '?', default=-1, 
            help = 'nuber of hidden units the model has. If not set it, inferred from the model path prefix.'
        )

        arg_parser.add_argument( '--beam_width', '-b',
            type = int, nargs = '?', default=10, 
            help = 'beam width used for stack decoding'
        )

        arg_parser.add_argument( '--nbest', '-n',
            type = int, nargs = '?', default=5,
            help = 'number of candidates (nbest) to predict per input name'
        )

        arg_parser.add_argument( '--alpha_to_kana', '-a2k',
            action="store_true",
            help = 'convert alphabets into kana'
        )

        arg_parser.add_argument( '--kana_to_alpha', '-k2a',
            action="store_true",
            help = 'convert kana to alphabets. If neither a2k or k2a is set, inferred from the model path prefix.'
        )

        return arg_parser

    def __str__(self):
        out_str = "PARAMS:\n"
        out_str += f"    model_path: [{self.model_path}]\n"
        if self.use_attention:
            out_str += f"    use_attention: [True]\n"
        else:
            out_str += f"    use_attention: [False]\n"
        out_str += f"    num_units: [{self.num_units}]\n"
        out_str += f"    beam_width: [{self.beam_width}]\n"
        out_str += f"    nbest: [{self.nbest}]\n"
        out_str += f"    max_length_alphas: [{self.max_length_alphas}]\n"
        out_str += f"    max_length_kanas: [{self.max_length_kanas}]\n"
        out_str += f"    tokenizer_path_alphas: [{self.tokenizer_path_alphas}]\n"
        out_str += f"    tokenizer_path_kanas: [{self.tokenizer_path_kanas}]\n"
        out_str += f"    direction: [{self.direction}]\n"
        return out_str


class TextCleander():

    def __init__(self):

        self.kanas_map  = cc.create_char_maps_kana()
        self.alphas_map = cc.create_char_maps_alpha()


    def clean_alpha(self, raw_alpha):
     
        cleaned_alpha = ""
        for c in raw_alpha.strip():
            if c not in self.alphas_map:
                return None
            else:
                cleaned_alpha += self.alphas_map[c]

        return cleaned_alpha
        
    def clean_kana(self, raw_kana):

        cleaned_kana = ""
        for c in raw_kana.strip():
            if c not in self.kanas_map:
                return None
            else:
                cleaned_kana += self.kanas_map[c]

        return cleaned_kana
        

class Encoder( tf.keras.Model ):

    def __init__( self, vocab_size, embedding_dim, enc_units ):

        super( Encoder, self ).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding( vocab_size, embedding_dim )
        self.gru = tf.keras.layers.GRU( self.enc_units,
                                        return_sequences = True,
                                        return_state = True,
                                        recurrent_initializer = 'glorot_uniform' )

    def call( self, x, hidden ):

        x = self.embedding(x)
        output, state = self.gru( x, initial_state = hidden )
        return output, state


class BahdanauAttention( tf.keras.layers.Layer ):

    def __init__( self, units ):
        super( BahdanauAttention, self ).__init__()

        self.W1 = tf.keras.layers.Dense( units )
        self.W2 = tf.keras.layers.Dense( units )
        self.V  = tf.keras.layers.Dense( 1 )


    def call( self, query, values ):

        query_with_time_axis = tf.expand_dims( query, 1 )

        score = self.V( tf.nn.tanh( self.W1( query_with_time_axis ) + self.W2( values ) ) )

        attention_weights = tf.nn.softmax( score, axis = 1 )

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum( context_vector, axis = 1 )

        return context_vector, attention_weights


class DecoderWithAttn(tf.keras.Model):

    def __init__( self, vocab_size, embedding_dim, dec_units ):
        super( DecoderWithAttn, self ).__init__()

        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding( vocab_size, embedding_dim )
        self.gru       = tf.keras.layers.GRU( self.dec_units,
                                              return_sequences = True,
                                              return_state = True,
                                              recurrent_initializer = 'glorot_uniform' )
        self.fc        = tf.keras.layers.Dense( vocab_size )
        self.attention = BahdanauAttention( self.dec_units )


    def call( self, x, hidden, enc_output ):

        context_vector, attention_weights = self.attention( hidden, enc_output )
        x             = self.embedding(x)
        x             = tf.concat( [tf.expand_dims(context_vector, 1), x], axis = -1 )
        output, state = self.gru(x)
        output        = tf.reshape( output, (-1, output.shape[2]) )
        x             = self.fc( output )

        return x, state, attention_weights


class DecoderWithoutAttn( tf.keras.Model ):

    def __init__( self, vocab_size, embedding_dim, dec_units ):
        super( DecoderWithoutAttn, self ).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding( vocab_size, embedding_dim )
        self.gru       = tf.keras.layers.GRU( self.dec_units,
                                              return_sequences = False,
                                              return_state = True,
                                              recurrent_initializer = 'glorot_uniform' )
        self.fc        = tf.keras.layers.Dense( vocab_size )

    def call( self, x, state ):
        x        = self.embedding(x)
        x, state = self.gru( x, state )
        x        = self.fc(x)
        return x, state


class ModelProvider():

    def __init__(self, cps): # cps : command line parameters server

        self.cps = cps

        with open(cps.tokenizer_path_alphas) as fa:
            data_a = json.load(fa)
            self.alphas_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data_a)

        with open(cps.tokenizer_path_kanas) as fk:
            data_k = json.load(fk)
            self.kanas_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data_k)

        if cps.direction == 'alpha_to_kana':
            self.emb_dim_encoder = len( self.alphas_tokenizer.word_index ) + 1
            self.emb_dim_decoder = len( self.kanas_tokenizer.word_index  ) + 1
            self.BOS = self.kanas_tokenizer.word_index['<']
            self.EOS = self.kanas_tokenizer.word_index['>']

        else:
            self.emb_dim_encoder = len( self.kanas_tokenizer.word_index  ) + 1
            self.emb_dim_decoder = len( self.alphas_tokenizer.word_index ) + 1
            self.BOS = self.alphas_tokenizer.word_index['<']
            self.EOS = self.alphas_tokenizer.word_index['>']

        self.encoder = Encoder( self.emb_dim_encoder, self.emb_dim_encoder, cps.num_units )

        if cps.use_attention:
            self.decoder = DecoderWithAttn( self.emb_dim_decoder, self.emb_dim_decoder, cps.num_units )
        else:
            self.decoder = DecoderWithoutAttn( self.emb_dim_decoder, self.emb_dim_decoder, cps.num_units )

        self.optimizer  = tf.keras.optimizers.Adam()

        self.checkpoint = tf.train.Checkpoint( optimizer = self.optimizer,
                                               encoder   = self.encoder,
                                               decoder   = self.decoder  )

        self.checkpoint.restore( cps.model_path ).expect_partial()

        self.stack_decoder = StackDecoder( self.decoder, self.BOS, self.EOS, use_attn = cps.use_attention )


    def evaluate( self, sentence ):
        if self.cps.direction == 'alpha_to_kana':
            return self.evaluate_alpha_to_kana(sentence)
        else:
            return self.evaluate_kana_to_alpha(sentence)


    def evaluate_alpha_to_kana(self, sentence):

        bracketed_sentence = '<' + sentence + '>'

        inputs = [ self.alphas_tokenizer.word_index[i] for i in bracketed_sentence ]

        inputs = tf.keras.preprocessing.sequence.pad_sequences( 
                     [inputs],
                     maxlen = self.cps.max_length_alphas,
                     padding = 'post'
                 )

        inputs = tf.convert_to_tensor(inputs)

        hidden = [ tf.zeros( ( 1, self.cps.num_units ) ) ]
        enc_out, enc_hidden = self.encoder( inputs, hidden )

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims( [self.kanas_tokenizer.word_index['<'] ], 0 )

        nbest_raw = self.stack_decoder.NBest( enc_out, enc_hidden, self.cps.beam_width, self.cps.nbest, self.cps.max_length_kanas + 2 ) 

        nbest_rtn = []
        for r in nbest_raw:
            result = ""
            for i in r.sentence[1:-1]:
                result += self.kanas_tokenizer.index_word[i] 
            nbest_rtn.append( (r.log_prob, result) )

        return nbest_rtn


    def evaluate_kana_to_alpha(self, sentence):

        bracketed_sentence = '<' + sentence + '>'

        inputs = [ self.kanas_tokenizer.word_index[i] for i in bracketed_sentence ]

        inputs = tf.keras.preprocessing.sequence.pad_sequences( 
                     [inputs],
                     maxlen = self.cps.max_length_kanas,
                     padding = 'post'
                 )

        inputs = tf.convert_to_tensor(inputs)

        hidden = [ tf.zeros( ( 1, self.cps.num_units ) ) ]
        enc_out, enc_hidden = self.encoder( inputs, hidden )

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims( [self.alphas_tokenizer.word_index['<'] ], 0 )

        nbest_raw = self.stack_decoder.NBest( enc_out, enc_hidden, self.cps.beam_width, self.cps.nbest, self.cps.max_length_alphas + 2 ) 

        nbest_rtn = []
        for r in nbest_raw:
            result = ""
            for i in r.sentence[1:-1]:
                result += self.alphas_tokenizer.index_word[i] 
            nbest_rtn.append( (r.log_prob, result) )

        return nbest_rtn


def main():

    cps = CommandParameterServer()
    print (cps)

    tc = TextCleander()

    model = ModelProvider(cps)

    line_no = 1
    for raw_in in cps.fp_in:
        if cps.direction == 'alpha_to_kana':
            cleaned_in = tc.clean_alpha(raw_in)
        else:
            cleaned_in = tc.clean_kana(raw_in)


        if cleaned_in and len(cleaned_in) > 0:

            print (f'processing valid line [{line_no}]')
            results = model.evaluate( cleaned_in )
            str_results = []
            for r in results:
                str_results.append( f'{math.exp(r[0]):.4f}\t{r[1]}' )

            cps.fp_out.write( raw_in.strip() + '\t' )
            cps.fp_out.write( '\t'.join(str_results) )
            cps.fp_out.write( '\n' )
        else:
            print(f'[{line_no}]: invalid name found in "{raw_in}"]')

        line_no += 1
    print (f'finished')


if __name__ == "__main__":
    main()
