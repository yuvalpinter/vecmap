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
from scipy.sparse import csr_matrix, csc_matrix, dia_matrix, identity, vstack, hstack
from scipy.sparse.linalg import inv, cg
from sklearn.linear_model import Lasso
from sklearn.decomposition import sparse_encode


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
    regsize = matr.shape[1]
    toinv = matr.transpose().dot(matr)
    precond = dia_matrix((1/(toinv.diagonal()),0),shape=(regsize,regsize))
    fullid = identity(regsize)
    return get_sparse_module(vstack([csr_matrix(cg(toinv, fullid.getrow(i).transpose().toarray(), maxiter=1, M=precond)[0]) \
                                    for i in range(regsize)]))
    
def psinv2(matr:csr_matrix, dtype, reg=0.):
    # inverse operation for sparse matrices doesn't seem to exist in cupy
    regsize = matr.shape[1]
    if reg == 0:
        regm = csr_matrix((regsize, regsize))
    else:
        regm = identity(regsize, dtype=dtype) * (1./reg)  # if direct cuda, call get_sparse_module()
    toinv = matr.transpose().dot(matr) + regm
    return get_sparse_module(inv(toinv))  # if direct cuda, add .get() to inv param


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
    recommended_type.add_argument('--toy', action='store_true', help='experiment with stuff on toy dataset')
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
    self_learning_group.add_argument('-v', '--verbose', action='store_true', help='write log information to stderr at each iteration')
    
    future_group = parser.add_argument_group('experimental arguments', 'Experimental arguments')
    future_group.add_argument('--trim_senses', action='store_true', help='Trim sense table to working vocab')
    future_group.add_argument('--lamb', type=float, default=0.5, help='Weight hyperparameter for sense alignment objectives')
    future_group.add_argument('--reglamb', type=float, default=1., help='Lasso regularization hyperparameter')
    future_group.add_argument('--inv_delta', type=float, default=0.0001, help='Delta_I added for inverting sense matrix')
    future_group.add_argument('--lasso_iters', type=int, default=10, help='Number of iterations for LASSO/NMF')
    
    args = parser.parse_args()

    # pre-setting groups
    if args.toy:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', vocabulary_cutoff=50, csls_neighborhood=10, trim_senses=True, inv_delta=1., reglamb=0.2, lasso_iters=100)
    if args.unsupervised or args.future:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', vocabulary_cutoff=2000, csls_neighborhood=10, trim_senses=True)
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
    print(f'source sense mapping of shape {src_senses.shape} loaded with {src_senses.getnnz()} nonzeros')
    
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
    
    src_size = x.shape[0] if args.vocabulary_cutoff <= 0 else min(x.shape[0], args.vocabulary_cutoff)
    trg_size = z.shape[0] if args.vocabulary_cutoff <= 0 else min(z.shape[0], args.vocabulary_cutoff)
    emb_dim = x.shape[1]
    
    if args.trim_senses:
        # reshape sense assignment
        src_senses = src_senses[:src_size]
        # new columns for words with no senses in original input
        ### TODO might also need this if not trimming (probably kinda far away)
        newcols = [csc_matrix(([1],([i],[0])),shape=(src_size,1)) for i in range(src_size)\
                   if src_senses.getrow(i).getnnz() == 0]
        # trim senses no longer used, add new ones
        colsums = src_senses.sum(axis=0).tolist()[0]
        src_senses = hstack([src_senses[:,[i for i,j in enumerate(colsums) if j>0]]] + newcols)
        print(f'trimmed sense dictionary dimensions: {src_senses.shape} with {src_senses.getnnz()} nonzeros')
    sense_size = src_senses.shape[1]
    
    # Initialize the concept embeddings from the source embeddings
    ### TODO maybe try gradient descent instead?
    ### TODO (pre-)create non-singular alignment matrix
    cc = xp.empty((sense_size, emb_dim))  # \tilde{E}
    print('starting psinv calc')
    t01 = time.time()
    src_sns_psinv = psinv(src_senses, dtype, args.inv_delta)
    xecc = x[:src_size].T.dot(get_sparse_module(src_senses).toarray()).T  # sense_size * emb_dim
    cc[:] = src_sns_psinv.dot(xecc)
    print(f'calculated psinv in {time.time()-t01:.2f} seconds', file=sys.stderr)
    
    ### TODO initialize trg_senses using seed dictionary instead?
    trg_sns_size = trg_size if args.trim_senses else z.shape[0]
    trg_senses = csr_matrix((trg_sns_size, sense_size))  # using non-cuda scipy because of 'inv' impl
    ccz = xp.empty_like(z)  # temp for trg_senses calculation
    zecc = xp.empty_like(xecc)  # sense_size * emb_dim
    
    # removed similarities memory assignment

    # Training loop
    lasso_model = Lasso(alpha=args.reglamb, fit_intercept=False, max_iter=args.lasso_iters,\
                        positive=True, warm_start=True)  # TODO more parametrization
    
    if args.log is not None:
        print(f'regularization: {args.reglamb}', file=log)
        print(f'lasso iterations: {args.lasso_iters}', file=log)
        print(f'inversion epsilon: {args.inv_delta}', file=log)
        log.flush()
    
    best_objective = objective = 10000.
    it = 1
    last_improvement = 0
    keep_prob = args.stochastic_initial
    t = time.time()
    end = False
    print('starting training')
    while True:
        if it % 50 == 0:
            print(f'starting iteration {it}, last objective was {objective}')

        # Increase the keep probability if we have not improved in args.stochastic_interval iterations
        if it - last_improvement > args.stochastic_interval:
            if keep_prob >= 1.0:
                end = True
            keep_prob = min(1.0, args.stochastic_multiplier*keep_prob)
            last_improvement = it
            
        ### update target assignments (6) - lasso regression
        time6 = time.time()
        # write to trg_senses (which should be sparse)
        # optimize: 0.5 * (xp.linalg.norm(zw[i] - trg_senses[i].dot(cc))^2) + (opts.reglamb * xp.linalg.norm(trg_senses[i],1))
        #print(zw[0] - (get_sparse_module(trg_senses[0]).dot(cc)))  # 1 * emb_dim
        
        # sparse_encode (not working on 20K vocab - "Killed"; ok on 2K)
        # n_samples === trg_sns_size; n_components === sense_size; n_features === emb_dim
        #sparseX = zw[:trg_size].get() # trg_sns_size, emb_dim
        #sparseD = cc.get() # sense_size, emb_dim TODO possibly pre-normalize
        #trg_senses = csr_matrix(sparse_encode(sparseX, sparseD, alpha=args.reglamb, max_iter=args.lasso_iters, positive=True))
        
        # lasso
        cccpu = cc.get().T  # emb_dim * sense_size
        
        # parallel lasso
        lasso_model.fit(cccpu, zw[:trg_size].get().T)
        trg_senses = csr_matrix(lasso_model.coef_)
        
        # non-parallel lasso
        #trgsnss = []
        #for i in range(trg_senses.shape[0]):
        #    if (i+1) % 10000 == 0:
        #        print(f'finished {i+1} lasso steps')
        #    lasso_model.fit(cccpu, zw[i].get())
        #    trgsnss.append(csr_matrix(lasso_model.coef_))
        #trg_senses = vstack(trgsnss)
        
        if args.verbose:
            print(f'sparse encoding step: {(time.time()-time6):.2f}', file=sys.stderr)
            if trg_senses.getnnz() > 0:
                print(f'finished target sense mapping step with {trg_senses.getnnz()} nonzeros.', file=sys.stderr)
        
        ### update synset embeddings (9)
        time9 = time.time()
        all_senses = vstack((src_senses, trg_senses), format='csr')
        all_sns_psinv = psinv(all_senses, dtype, args.inv_delta)
        xzecc = xp.concatenate((xw[:src_size], zw[:trg_size])).T\
                    .dot(get_sparse_module(all_senses).toarray()).T  # sense_size * emb_dim
        cc[:] = all_sns_psinv.dot(xzecc)
        if args.verbose:
            print(f'synset embedding update: {time.time()-time9:.2f}', file=sys.stderr)
        
        ### update projections (3,5)
        # write to zw and xw
        # xecc is constant
        if args.orthogonal or not end:
            time3 = time.time()
            u, s, vt = xp.linalg.svd(cc.T.dot(xecc))
            wx = vt.T.dot(u.T).astype(dtype)
            x.dot(wx, out=xw)
            if args.verbose:
                print(f'source projection update: {time.time()-time3:.2f}', file=sys.stderr)
            
            time3 = time.time()
            zecc = z[:trg_size].T.dot(get_sparse_module(trg_senses).toarray()).T
            u, s, vt = xp.linalg.svd(cc.T.dot(zecc))
            wz = vt.T.dot(u.T).astype(dtype)
            z.dot(wz, out=zw)
            if args.verbose:
                print(f'target projection update: {time.time()-time3:.2f}', file=sys.stderr)
            
        else:  # advanced mapping
            break ### TODO strip for parts

            # remove lower-rank transformations
            midpoint = src_size * args.max_align
            src_indices = xp.concatenate((src_indices[:src_size], src_indices[midpoint:midpoint+trg_size]))
            trg_indices = xp.concatenate((trg_indices[:src_size], trg_indices[midpoint:midpoint+trg_size]))
            
            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z
            
            ### TODO entry point for adding more matrix operations ###

            # STEP 1: Whitening
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
            time_obj = time.time()
            objective = (xp.linalg.norm(xw[:src_size] - get_sparse_module(src_senses).dot(cc),'fro')\
                            + xp.linalg.norm(zw[:trg_size] - get_sparse_module(trg_senses).dot(cc),'fro')) / 2 \
                        + args.reglamb * trg_senses.sum()  # TODO consider thresholding reg part
            objective = float(objective)
            if args.verbose:
                print(f'objective calculation: {time.time()-time_obj:.2f}', file=sys.stderr)  # 0.020 sec
            ### TODO create bilingual dictionary?
            
            if objective - best_objective <= -args.threshold:
                last_improvement = it
                best_objective = objective

            # Logging
            duration = time.time() - t
            if args.verbose:
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('\t- Objective:        {0:9.4f}'.format(objective), file=sys.stderr)
                print('\t- Drop probability: {0:9.4f}%'.format(100 - 100*keep_prob), file=sys.stderr)
                print(file=sys.stderr)
                sys.stderr.flush()
            if args.log is not None:
                print(f'{it}\t{objective:.6f}\t{duration:.6f}\t{trg_senses.getnnz()}', file=log)
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
