# Copyright (C) 2019 Yuval Pinter <yuvalpinter@gmail.com>
#               2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import re
import sys
import time
import pickle
from scipy.sparse import csr_matrix


def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m*mask


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('sense_input', help='the input sense mapping matrix')
    parser.add_argument('src_output', help='the output source embeddings')
    parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('dict_output', default='dictionary.pkl', help='the output dictionary pickle file')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')

    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--supervised', metavar='DICTIONARY', help='recommended if you have a large training dictionary')
    recommended_type.add_argument('--semi_supervised', metavar='DICTIONARY', help='recommended if you have a small seed dictionary')
    recommended_type.add_argument('--identical', action='store_true', help='recommended if you have no seed dictionary but can rely on identical words')
    recommended_type.add_argument('--unsupervised', action='store_true', help='recommended if you have no seed dictionary and do not want to rely on identical words')
    recommended_type.add_argument('--future', action='store_true', help='experiment with stuff')
    recommended_type.add_argument('--acl2018', action='store_true', help='reproduce our ACL 2018 system')

    init_group = parser.add_argument_group('advanced initialization arguments', 'Advanced initialization arguments')
    init_type = init_group.add_mutually_exclusive_group()
    init_type.add_argument('-d', '--init_dictionary', default=sys.stdin.fileno(), metavar='DICTIONARY', help='the training dictionary file (defaults to stdin)')
    init_type.add_argument('--init_identical', action='store_true', help='use identical words as the seed dictionary')
    init_type.add_argument('--init_numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    init_type.add_argument('--init_unsupervised', action='store_true', help='use unsupervised initialization')
    init_group.add_argument('--unsupervised_vocab', type=int, default=0, help='restrict the vocabulary to the top k entries for unsupervised initialization')

    mapping_group = parser.add_argument_group('advanced mapping arguments', 'Advanced embedding mapping arguments')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=[], help='the normalization actions to perform in order')
    mapping_group.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    mapping_group.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the source language embeddings')
    mapping_group.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the target language embeddings')
    mapping_group.add_argument('--src_dewhiten', choices=['src', 'trg'], help='de-whiten the source language embeddings')
    mapping_group.add_argument('--trg_dewhiten', choices=['src', 'trg'], help='de-whiten the target language embeddings')
    mapping_group.add_argument('--dim_reduction', type=int, default=0, help='apply dimensionality reduction')
    mapping_type = mapping_group.add_mutually_exclusive_group()
    mapping_type.add_argument('-c', '--orthogonal', action='store_true', help='use orthogonal constrained mapping')
    
    future_group = parser.add_argument_group('experimental arguments', 'Experimental arguments')
    future_group.add_argument('--max_align', type=int, default=1, help='Number of top-ranked elements to align to each word (defaults to 1=base)')
    future_group.add_argument('--align_weight', choices=['unit', 'rr', 'softmax'], default='rr', help='Weights assigned to ranked elements in maximization phase (unit - no weighting; rr - reciprocal rank; softmax - NOT IMPLEMENTED YET)')
    future_group.add_argument('--lamb', type=float, default=0.5, help='Weight hyperparameter for sense alignment objectives')
    future_group.add_argument('--sense_init', choices=['none', 'glorot'], default='glorot')
    
    self_learning_group = parser.add_argument_group('advanced self-learning arguments', 'Advanced arguments for self-learning')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--vocabulary_cutoff', type=int, default=0, help='restrict the vocabulary to the top k entries')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='the direction for dictionary induction (defaults to union)')
    self_learning_group.add_argument('--csls', type=int, nargs='?', default=0, const=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, metavar='DICTIONARY', help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--stochastic_initial', default=0.1, type=float, help='initial keep probability stochastic dictionary induction (defaults to 0.1)')
    self_learning_group.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    self_learning_group.add_argument('--stochastic_interval', default=50, type=int, help='stochastic dictionary induction interval (defaults to 50)')
    self_learning_group.add_argument('--log', default='map.log', help='write to a log file in tsv format at each iteration')
    self_learning_group.add_argument('-v', '--verbose', action='store_true', help='write log information to stderr at each iteration')
    args = parser.parse_args()

    # pre-setting groups
    if args.supervised is not None:
        parser.set_defaults(init_dictionary=args.supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.semi_supervised is not None:
        parser.set_defaults(init_dictionary=args.semi_supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.identical:
        parser.set_defaults(init_identical=True, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    
    if args.unsupervised or args.future:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10, max_align=2, align_weight='rr')
    if args.unsupervised or args.acl2018:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    args = parser.parse_args()

    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'  # many operations not supported by cupy
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    print('reading embeddings...')
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    print('embeddings read')
    
    # Read input source sense mapping
    print('reading sense mapping')
    src_senses = pickle.load(open(args.sense_input, 'rb'))
    if src_senses.shape[0] != x.shape[0]:
        src_senses = csr_matrix(src_senses.transpose())
    src_senses = get_sparse_module(src_senses)
    print(f'source sense mapping of shape {src_senses.shape} loaded')
    
    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
        print('CUDA loaded')
    else:
        xp = np
    xp.random.seed(args.seed)

    # Build word to index map (only relevant in supervised learning or with validation)
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    print(f'mapped {len(src_words)} source words')
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    print(f'mapped {len(trg_words)} target words')

    # STEP 0: Normalization
    embeddings.normalize(x, args.normalize)
    embeddings.normalize(z, args.normalize)
    print('normalization complete')

    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    if args.init_unsupervised:
        # default, with unsupervised_vocab = top 4k aligned words
        sim_size = min(x.shape[0], z.shape[0]) if args.unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0], args.unsupervised_vocab)
        u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u*s).dot(u.T)
        u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, args.normalize)
        embeddings.normalize(zsim, args.normalize)
        sim = xsim.dot(zsim.T)
        if args.csls_neighborhood > 0:
            knn_sim_fwd = topk_mean(sim, k=args.csls_neighborhood)
            knn_sim_bwd = topk_mean(sim.T, k=args.csls_neighborhood)
            sim -= knn_sim_fwd[:, xp.newaxis]/2 + knn_sim_bwd/2
        if args.direction == 'forward':
            src_indices = xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif args.direction == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = xp.arange(sim_size)
        elif args.direction == 'union':
            src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
            trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
        del xsim, zsim, sim
        print(f'initialized unsupervised dictionary')
    elif args.init_numerals:
        numeral_regex = re.compile('^[0-9]+$')
        src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
        trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)
        for word in numerals:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
        print('initialized numeral dictionary')
    elif args.init_identical:
        identical = set(src_words).intersection(set(trg_words))
        for word in identical:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
        print('initialized identical dictionary')
    else:
        f = open(args.init_dictionary, encoding=args.encoding, errors='surrogateescape')
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)
        f.close()
        print('initialized seed dictionary')

    # Read validation dictionary
    if args.validation is not None:
        f = open(args.validation, encoding=args.encoding, errors='surrogateescape')
        validation = collections.defaultdict(set)
        oov = set()
        vocab = set()
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                validation[src_ind].add(trg_ind)
                vocab.add(src)
            except KeyError:
                oov.add(src)
        oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
        validation_coverage = len(validation) / (len(validation) + len(oov))
        print(f'loaded validation dictionary with {validation_coverage:.3f} coverage')

    # Create log file
    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')
        print(f'logging into {args.log}')

    ### Allocate memory ###
    # W matrices
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)
    src_size = x.shape[0] if args.vocabulary_cutoff <= 0 else min(x.shape[0], args.vocabulary_cutoff)
    trg_size = z.shape[0] if args.vocabulary_cutoff <= 0 else min(z.shape[0], args.vocabulary_cutoff)
    
    emb_dim = x.shape[1]
    sense_size = src_senses.shape[1]
    
    ### TODO initialize using seed dictionary
    trg_senses = get_sparse_module(csr_matrix((trg_size, sense_size)))
    
    # sense embeddings and alignments
    # src_s_size * sdim embedding table (\tilde{E}^h)
    if args.sense_init == 'glorot':
        sx = xp.random.rand(sense_size, emb_dim)*xp.sqrt(1/(sense_size+emb_dim))
        sz = xp.random.rand(sense_size, emb_dim)*xp.sqrt(1/(sense_size+emb_dim))
    else:
        sx = xp.empty((sense_size, emb_dim), dtype=dtype)
        sz = xp.empty((sense_size, emb_dim), dtype=dtype)
    
    ### TODO these could be svd'd? Do these replace xw, zw?
    ccx = xp.empty_like(x)  # temp for least-squares calc
    ccz = xp.empty_like(z)  # temp for least-squares calc
    
    # similarities for ranking
    simfwd = xp.empty((min(src_size, args.batch_size), trg_size), dtype=dtype)
    simbwd = xp.empty((min(trg_size, args.batch_size), src_size), dtype=dtype)
    xr = xp.zeros(((src_size+trg_size) * args.max_align, x.shape[1]), dtype=dtype)  # assumes "both" param
    zr = xp.zeros(((src_size+trg_size) * args.max_align, z.shape[1]), dtype=dtype)  # assumes "both" param
    
    # similarity weight coefficients
    all_coefs = xp.zeros(((src_size+trg_size) * args.max_align, 1), dtype=dtype)
    
    # top-ranked similarities
    argsimsf = xp.empty((min(src_size, args.batch_size), 1), dtype=int)
    argsimsb = xp.empty((min(trg_size, args.batch_size), 1), dtype=int)
    if args.validation is not None:
        simval = xp.empty((len(validation.keys()), z.shape[0]), dtype=dtype)
    best_sim_forward = xp.full(src_size, -100, dtype=dtype)

    # nonzero elements in dictionaries
    src_indices_forward = xp.array(list(range(src_size)) * args.max_align)
    trg_indices_forward = xp.zeros(src_size * args.max_align, dtype=int)
    best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
    src_indices_backward = xp.zeros(trg_size * args.max_align, dtype=int)
    trg_indices_backward = xp.array(list(range(trg_size)) * args.max_align)
    
    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)

    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = args.stochastic_initial
    t = time.time()
    end = not args.self_learning
    print('starting training')
    while True:
        if it % 50 == 0:
            print(f'starting iteration {it}')

        # Increase the keep probability if we have not improved in args.stochastic_interval iterations
        if it - last_improvement > args.stochastic_interval:
            if keep_prob >= 1.0:
                end = True
            keep_prob = min(1.0, args.stochastic_multiplier*keep_prob)
            last_improvement = it
            
        ### UNDER CONSTRUCTION - SENSE EMBEDDING PHASE ### 
        ### TODO move to appropriate location? Replace dictionary learning?
        
        # src_senses is a *fixed* src_size * src_s_size sparse matrix (S^h in paper)
        # sx is a src_s_size * sdim embedding table (\tilde{E}^h)
        # sz is a src_s_size * sdim embedding table (\tilde{E}^l)
        # lamb is a regularization hyperparam from input
        
        ccx = xw - src_senses.dot(sz)  # Y in eq. (10), shape = src_size * dim
        del_sx_1 = xp.linalg.inv(src_senses.dot(src_senses.transpose())+(xp.identity(src_size)*(1./args.lamb)))  # shape = src_size * src_size PROBLEM
        del_sx_2 = src_senses.transpose().dot(del_sx_1)  # changed order here for cupy reasons, might want to change back
        del_sx = del_sx_2.dot(ccx)  # eq. (11)
        sx[:] = sz + del_sx
        
        ccz = zw - trg_senses.dot(sx)  # Y in eq. (10) but for target, shape = trg_size * dim
        del_sz_1 = xp.linalg.inv(trg_senses.dot(trg_senses.transpose())+(xp.identity(trg_size)*(1./args.lamb)))  # shape = trg_size * trg_size PROBLEM
        del_sz_2 = trg_senses.transpose().dot(del_sz_1)  # changed order here for cupy reasons, might want to change back
        del_sz = del_sz_2.dot(ccz)  # eq. (11)
        sz[:] = sx + del_sz
        
        ### OLD FORM - might still be true
        #ccz = zw - trg_senses.dot(sx) # Y in eq. (10) but for target
        #del_sz = xp.linalg.inv(trg_senses.dot(trg_senses.transpose())+(xp.identity(trg_size)*(1./args.lamb)))\
        #            .dot(trg_senses.transpose()).dot(ccz)  # eq. (11) but for target
        #sz[:] = sx + del_sz
        
        ### TODO missing trg_senses optimization phase (S^l in paper; src_senses is fixed)
        
        ### END CONSTRUCTION - SENSE EMBEDDING PHASE ###

        # Update the embedding mapping (only affecting vectors that have dictionary mappings)
        if args.orthogonal or not end:  # orthogonal mapping
            if it == 1:
                # only initialized alignment available
                u, s, vt = xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
            else:
                if args.align_weight == 'softmax':
                    ### TODO individualized softmax coefficients ###
                    raise 'Softmax weights not supported yet'           
                else:
                    ### TODO I'm assuming here that the alignment method is 'both', so everything's double
                    ### TODO all_coefs can be computed outside the iteration loop
                    # format: src_size_0, ..., src_size_k-1, trg_size_0, ..., trg_size_k-1
                    ncopies = args.max_align
                    cutoffs = list(range(src_size*ncopies)[::src_size]) \
                              + list(range(src_size*ncopies,(src_size+trg_size)*ncopies)[::trg_size])
                    if args.align_weight == 'rr':
                        coefs = [1. / (k+1) for k in range(ncopies)] * 2            
                    else:  # 'unit'
                        coefs = [1.] * (ncopies * 2)
                    for cf, co_s, co_e in zip(coefs, cutoffs, cutoffs[1:] + [len(all_coefs)]):
                        all_coefs[co_s:co_e] = cf
                    zr = z[trg_indices] * all_coefs
                    xr = x[src_indices] * all_coefs
                    u, s, vt = xp.linalg.svd(zr.T.dot(xr))
            w = vt.T.dot(u.T)
            x.dot(w, out=xw)
            zw[:] = z
        else:  # advanced mapping (default for end, acl2018)

            # remove lower-rank transformations
            midpoint = src_size * args.max_align
            src_indices = xp.concatenate((src_indices[:src_size], src_indices[midpoint:midpoint+trg_size]))
            trg_indices = xp.concatenate((trg_indices[:src_size], trg_indices[midpoint:midpoint+trg_size]))
            
            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z
            
            ### TODO entry point for adding more matrix operations ###

            # STEP 1: Whitening
            ### TODO figure out how weighted k-best affects this (and onwards) ###
            def whitening_transformation(m):
                u, s, vt = xp.linalg.svd(m, full_matrices=False)
                return vt.T.dot(xp.diag(1/s)).dot(vt)
            if args.whiten:
                wx1 = whitening_transformation(xw[src_indices])
                wz1 = whitening_transformation(zw[trg_indices])
                xw = xw.dot(wx1)
                zw = zw.dot(wz1)

            # STEP 2: Orthogonal mapping
            wx2, s, wz2_t = xp.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
            wz2 = wz2_t.T
            xw = xw.dot(wx2)
            zw = zw.dot(wz2)

            # STEP 3: Re-weighting
            xw *= s**args.src_reweight
            zw *= s**args.trg_reweight

            # STEP 4: De-whitening
            if args.src_dewhiten == 'src':
                xw = xw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.src_dewhiten == 'trg':
                xw = xw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))
            if args.trg_dewhiten == 'src':
                zw = zw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.trg_dewhiten == 'trg':
                zw = zw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))

            # STEP 5: Dimensionality reduction (default: OFF (0))
            if args.dim_reduction > 0:
                xw = xw[:, :args.dim_reduction]
                zw = zw[:, :args.dim_reduction]

        # Self-learning
        if end:
            break
        else:
            # Update the training dictionary (default direction - union)
            if args.direction in ('forward', 'union'):
                if args.csls_neighborhood > 0:  # default acl2018: 10
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)  # get next batch to operate on
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                    simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN
                    
                    # softmaxing
                    for k in range(args.max_align):
                        argsimsf = dropout(simfwd[:j-i], 1 - keep_prob).argmax(axis=1)
                        simfwd[:j-i,argsimsf] = -200
                        trg_indices_forward[(k*src_size)+i:(k*src_size)+j] = argsimsf
            if args.direction in ('backward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)  # get next batch to operate on
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                    simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j-i] -= knn_sim_fwd/2  # Equivalent to the real CSLS scores for NN
                    
                    # softmaxing
                    for k in range(args.max_align):
                        argsimsb = dropout(simbwd[:j-i], 1 - keep_prob).argmax(axis=1)
                        simbwd[:j-i,argsimsb] = -200
                        trg_indices_backward[(k*trg_size)+i:(k*trg_size)+j] = argsimsb
            if args.direction == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif args.direction == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif args.direction == 'union':
                src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            if args.direction == 'forward':
                objective = xp.mean(best_sim_forward).tolist()
            elif args.direction == 'backward':
                objective = xp.mean(best_sim_backward).tolist()
            elif args.direction == 'union':  # default
                objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= args.threshold:
                last_improvement = it
                best_objective = objective

            # Accuracy and similarity evaluation in validation (default - off)
            if args.validation is not None:
                src = list(validation.keys())
                xw[src].dot(zw.T, out=simval)
                nn = asnumpy(simval.argmax(axis=1))
                accuracy = np.mean([1 if nn[i] in validation[src[i]] else 0 for i in range(len(src))])
                similarity = np.mean([max([simval[i, j].tolist() for j in validation[src[i]]]) for i in range(len(src))])

            # Logging
            duration = time.time() - t
            if args.verbose:
                print(file=sys.stderr)
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
                print('\t- Drop probability: {0:9.4f}%'.format(100 - 100*keep_prob), file=sys.stderr)
                if args.validation is not None:
                    print('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity), file=sys.stderr)
                    print('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy), file=sys.stderr)
                    print('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage), file=sys.stderr)
                sys.stderr.flush()
            if args.log is not None:
                val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
                    100 * similarity, 100 * accuracy, 100 * validation_coverage) if args.validation is not None else ''
                print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=log)
                log.flush()

        t = time.time()
        it += 1

    # Write mapped embeddings
    srcfile = open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(src_words, xw, srcfile)
    embeddings.write(trg_words, zw, trgfile)
    srcfile.close()
    trgfile.close()
    
    # Write dictionary
    dictfile = open(args.dict_output, mode='wb')
    dictalign = list(zip(src_indices, trg_indices))
    pickle.dump(dictalign, dictfile)


if __name__ == '__main__':
    main()
