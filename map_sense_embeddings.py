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
from scipy.sparse import csr_matrix, identity, vstack
from scipy.sparse.linalg import inv


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

    
def psinv(matr:csr_matrix, dtype, reg=0.):
    # inverse operation for sparse matrices doesn't seem to exist in cupy
    regsize = matr.shape[1]
    if reg == 0:
        regm = csr_matrix((regsize, regsize))
    else:
        regm = identity(regsize, dtype=dtype) * (1./reg)  # if direct cuda, call get_sparse_module()
    return get_sparse_module(inv(matr.transpose().dot(matr) + regm))  # if direct cuda, add .get() to inv param


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('sense_input', help='the input sense mapping matrix')
    parser.add_argument('src_output', help='the output source embeddings')
    parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('tsns_output', default='tsns.pkl', help='the output target senses pickle file')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')

    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--unsupervised', action='store_true', help='recommended if you have no seed dictionary and do not want to rely on identical words')
    recommended_type.add_argument('--future', action='store_true', help='experiment with stuff')
    recommended_type.add_argument('--acl2018', action='store_true', help='reproduce our ACL 2018 system')

    init_group = parser.add_argument_group('advanced initialization arguments', 'Advanced initialization arguments')
    init_type = init_group.add_mutually_exclusive_group()
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
    
    self_learning_group = parser.add_argument_group('advanced self-learning arguments', 'Advanced arguments for self-learning')
    self_learning_group.add_argument('--vocabulary_cutoff', type=int, default=0, help='restrict the vocabulary to the top k entries')
    self_learning_group.add_argument('--csls', type=int, nargs='?', default=0, const=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--stochastic_initial', default=0.1, type=float, help='initial keep probability stochastic dictionary induction (defaults to 0.1)')
    self_learning_group.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    self_learning_group.add_argument('--stochastic_interval', default=50, type=int, help='stochastic dictionary induction interval (defaults to 50)')
    self_learning_group.add_argument('--log', default='map.log', help='write to a log file in tsv format at each iteration')
    
    future_group = parser.add_argument_group('experimental arguments', 'Experimental arguments')
    future_group.add_argument('--lamb', type=float, default=0.5, help='Weight hyperparameter for sense alignment objectives')
    future_group.add_argument('--reglamb', type=float, default=0.01, help='Lasso regularization hyperparameter')
    
    args = parser.parse_args()

    # pre-setting groups    
    if args.unsupervised or args.future:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.unsupervised or args.acl2018:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', vocabulary_cutoff=20000, csls_neighborhood=10)
    args = parser.parse_args()

    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'  # many operations not supported by cupy
    elif args.precision == 'fp32':  # default
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
        src_senses = csr_matrix(src_senses.transpose())  # using non-cuda scipy because of 'inv' impl
    #src_senses = get_sparse_module(src_senses)
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

    # removed word to index map (only relevant in supervised learning or with validation)

    # STEP 0: Normalization
    embeddings.normalize(x, args.normalize)
    embeddings.normalize(z, args.normalize)
    print('normalization complete')

    # removed building the seed dictionary

    # removed validation step

    # Create log file
    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')
        print(f'logging into {args.log}')    

    # Allocate memory
    
    # Initialize the projection matrices W(s) = W(t) = I.
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)
    xw[:] = x
    zw[:] = z
    
    ### TODO remove? or, possibly, trim sense table?
    src_size = x.shape[0] if args.vocabulary_cutoff <= 0 else min(x.shape[0], args.vocabulary_cutoff)
    trg_size = z.shape[0] if args.vocabulary_cutoff <= 0 else min(z.shape[0], args.vocabulary_cutoff)
    
    emb_dim = x.shape[1]
    sense_size = src_senses.shape[1]
    
    # Initialize the concept embeddings from the source embeddings
    ### TODO maybe try gradient descent instead?
    ### TODO (pre-)create non-singular alignment matrix
    cc = xp.empty((sense_size, emb_dim))  # \tilde{E}
    src_sns_psinv = psinv(src_senses, dtype, 0.001)  # will come in handy in iterations
    xecc = x.T.dot(get_sparse_module(src_senses).toarray()).T  # sense_size * emb_dim
    cc[:] = src_sns_psinv.dot(xecc)
        
    ### TODO initialize trg_senses using seed dictionary instead?
    #trg_senses = csr_matrix((trg_size, sense_size))  # uncomment if trimming src_senses
    trg_senses = csr_matrix((z.shape[0], sense_size))  # using non-cuda scipy because of 'inv' impl
    ccz = xp.empty_like(z)  # temp for trg_senses calculation
    zecc = xp.empty_like(xecc)  # sense_size * emb_dim
    
    # removed similarities memory assignment

    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = args.stochastic_initial
    t = time.time()
    end = False
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
            
        ### TODO update target assignments (6) - lasso regression
        # write to trg_senses (which should be sparse)
        # optimize: 0.5 * (xp.linalg.norm(zw[i] - trg_senses[i].dot(cc))^2) + (opts.reglamb * xp.linalg.norm(trg_senses[i],1))
        #print(zw[0] - (get_sparse_module(trg_senses[0]).dot(cc)))  # 1 * emb_dim
        
        ### update synset embeddings (9)
        ### TODO probably no memory for this
        all_senses = vstack((src_senses, trg_senses), format='csr')
        all_sns_psinv = psinv(all_senses, dtype, 0.001)
        xzecc = xp.concatenate((xw, zw)).T.dot(get_sparse_module(all_senses).toarray()).T  # sense_size * emb_dim
        cc[:] = all_sns_psinv.dot(xzecc)
        
        ### update projections (3,5)
        # write to zw and xw
        # xecc is constant
        if args.orthogonal or not end:
            u, s, vt = xp.linalg.svd(cc.T.dot(xecc))
            wx = vt.T.dot(u.T).astype(dtype)
            x.dot(wx, out=xw)
            
            zecc = z.T.dot(get_sparse_module(trg_senses).toarray()).T
            u, s, vt = xp.linalg.svd(cc.T.dot(zecc))
            wz = vt.T.dot(u.T).astype(dtype)
            z.dot(wz, out=zw)
            
        else:  # advanced mapping
            pass ### TODO strip for parts

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
        
        if end:
            break
        else:
            if False: ### TODO strip for parts
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
                src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

            ### TODO compute the objective, check against a stopping condition 
            # Objective function evaluation
            objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= args.threshold:
                last_improvement = it
                best_objective = objective

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
    with open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape') as srcfile:
        embeddings.write(src_words, xw, srcfile)
    with open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape') as trgfile:
        embeddings.write(trg_words, zw, trgfile)
    
    # Write target sense embeddings
    with open(args.tsns_output, mode='wb') as tsnsfile:
        pickle.dump(trg_senses, tsnsfile)

if __name__ == '__main__':
    main()
