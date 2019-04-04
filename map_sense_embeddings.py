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
#from heap import Heap
from cupy_utils import *
#from learning import SGD, Adam

import argparse
import collections
import numpy as np
import re
import sys
import time
import pickle
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, dia_matrix, identity, vstack, hstack
from scipy.sparse.linalg import inv, cg
from sklearn.linear_model import Lasso


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

    
def sparse_id(n):
    return dia_matrix(([1]*n,0),shape=(n, n))


def psinv(matr, dtype, reg=0.):
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


def batch_sparse(a, batch=500):
    ### TODO increase default batch if allowed
    sparses_to_stack = []
    for i in range(0, a.shape[0], batch):
        end = min(i+batch, a.shape[0])
        if end == i: break
        sparses_to_stack.append(csr_matrix(a[i:end].get()))
    return vstack(sparses_to_stack)
    
    
def trim_sparse(a, k, issparse=False, clip=None):
    '''
    Return a sparse matrix with all but top k values zeros
    TODO ensure 1 nonzero per row + per column
    TODO clip instead of scaling
    '''
    if issparse:
        if a.getnnz() <= k:
            return a
        kth_quant = 100 * (1. - (k / a.getnnz()))
        xp = get_array_module(a.data)
        kth = xp.percentile(a.data, kth_quant, interpolation='lower')
        mask = a.data > kth
        a.data = a.data * mask
        a.eliminate_zeros()
        if clip is not None:
            a.data.clip(-clip, clip, out=a.data)
        return a
    else:
        maxval = a.max()
        val = maxval / 10
        mask = a > val
        while sum(sum(mask)) > k:
            val *= 1.25  # 10 searches max; with 1.5 it's 5
            if val >= 1.0:
                break
            mask = a > val
        a *= mask
        sprs = batch_sparse(a)
        if clip is not None:
            sprs.data.clip(-clip, clip, out=sprs.data)
        return sprs
    

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
    future_group.add_argument('--iterations', type=int, default=-1, help='Number of overall model iterations')
    future_group.add_argument('--trg_batch', type=int, default=5000, help='Batch size for target steps')
    future_group.add_argument('--gd', action='store_true', help='Apply gradient descent for assignment and synset embeddings')
    future_group.add_argument('--gd_lr', type=float, default=1e-2, help='Learning rate for SGD (default=0.01)')
    future_group.add_argument('--gd_clip', type=float, default=5., help='Per-coordinate gradient clipping (default=5)')
    future_group.add_argument('--gd_emb_steps', type=int, default=1, help='Consecutive steps for each sense embedding update phase')
    future_group.add_argument('--base_prox_lambda', type=float, default=0.99, help='Lambda for proximal gradient in lasso step')
    future_group.add_argument('--prox_decay', action='store_true', help='Multiply proximal lambda by itself each iteration')
    future_group.add_argument('--sense_limit', type=float, default=1.1, help='maximum amount of target sense mappings, in terms of source mappings (default=1.1x)')
    
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
    print(f'initialized concept embeddings in {time.time()-t01:.2f} seconds', file=sys.stderr)
    if args.verbose:
        # report precision of psedo-inverse operation, checked by inverting
        pseudo_id = src_senses.transpose().dot(src_senses).dot(src_sns_psinv.get())
        real_id = sparse_id(sense_size)
        rel_diff = (pseudo_id-real_id).sum() / (sense_size*sense_size)
        print(f'per-coordinate pseudo-inverse precision is {rel_diff:.5f}')
    
    ### TODO initialize trg_senses using seed dictionary instead?
    trg_sns_size = trg_size if args.trim_senses else z.shape[0]
    trg_senses = csr_matrix((trg_sns_size, sense_size))  # using non-cuda scipy because of 'inv' impl
    zecc = xp.empty_like(xecc)  # sense_size * emb_dim
    #tg_grad = xp.empty((trg_sns_size, sense_size))
    
    if args.gd:
        # everything can be done on gpu
        src_senses = get_sparse_module(src_senses)
        trg_senses = get_sparse_module(trg_senses)
        if args.sense_limit > 0.0:
            trg_sense_limit = int(args.sense_limit * src_senses.getnnz())
            if args.verbose:
                print(f'limiting target side to {trg_sense_limit} sense mappings')
        else:
            trg_sense_limit = -1
    
    ### TODO return memory assignment for similarities?

    # Training loop
    if args.gd:
        prox_lambda = args.base_prox_lambda
        
        ### TODO sgd model not currently used
        ### if uncommented, remember to commit learning.py
        #sgd_model = SGD(args.gd_lr)
    else:
        lasso_model = Lasso(alpha=args.reglamb, fit_intercept=False, max_iter=args.lasso_iters,\
                            positive=True, warm_start=True)  # TODO more parametrization
    
    if args.log is not None:
        if args.gd:
            print(f'gradient descent lr: {args.gd_lr}', file=log)
        else:
            print(f'lasso regularization: {args.reglamb}', file=log)
            print(f'lasso iterations: {args.lasso_iters}', file=log)
            print(f'inversion epsilon: {args.inv_delta}', file=log)
        log.flush()
    
    best_objective = objective = 1000000.
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
        
        if args.iterations > 0 and it > args.iterations:
            end=True
            
        ### update target assignments (6) - lasso-esque regression
        time6 = time.time()            
        # write to trg_senses (which should be sparse)
        # optimize: 0.5 * (xp.linalg.norm(zw[i] - trg_senses[i].dot(cc))^2) + (opts.reglamb * xp.linalg.norm(trg_senses[i],1))
        #print(zw[0] - (get_sparse_module(trg_senses[0]).dot(cc)))  # 1 * emb_dim
        
        if args.gd:
            ### TODO use sgd_model?
            ### TODO enforce nonnegativity
            ### TODO handle sizes and/or threshold sparse matrix - possibly by batching vocab
            # st <- st + eta * (ew - st.dot(es)).dot(es.T)
            # allow up to sense_limit updates, clip gradient
            
            batch_grads = []
            for i in range(0, trg_size, args.trg_batch):
                batch_end = min(i+args.trg_batch, trg_size)
                tg_grad_b = (zw[i:batch_end] - trg_senses[i:batch_end].dot(cc)).dot(cc.T)
                
                # proximal gradient
                tg_grad_b -= prox_lambda
                tg_grad_b.clip(0.0, out=tg_grad_b)
                batch_grads.append(batch_sparse(tg_grad_b))
                
            tg_grad = get_sparse_module(vstack(batch_grads))
            del tg_grad_b
            
            if args.prox_decay:
                prox_lambda *= args.base_prox_lambda
            
            trg_senses += args.gd_lr * tg_grad
            
            # allow up to sense_limit nonzeros
            if trg_sense_limit > 0:
                trg_senses = trim_sparse(trg_senses, trg_sense_limit, issparse=True, clip=None)
            
            ### TODO consider finishing up with lasso (maybe only in final iteration)
            
        else:
            # parallel LASSO (no cuda impl)
            cccpu = cc.get().T  # emb_dim * sense_size
            lasso_model.fit(cccpu, zw[:trg_size].get().T)
            ### TODO maybe trim, keep only above some threshold (0.05) OR top f(#it)
            trg_senses = lasso_model.sparse_coef_
        
        if args.verbose:
            print(f'target sense mapping step: {(time.time()-time6):.2f}, {trg_senses.getnnz()} nonzeros', file=sys.stderr)
            objective = ((xp.linalg.norm(xw[:src_size] - get_sparse_module(src_senses).dot(cc),'fro') ** 2)\
                            + (xp.linalg.norm(zw[:trg_size] - get_sparse_module(trg_senses).dot(cc),'fro')) ** 2) / 2 \
                        + args.reglamb * trg_senses.sum()  # TODO consider thresholding reg part
            objective = float(objective)
            print(f'objective: {objective:.3f}')
        
        # Write target sense mapping (no time issue)
        with open('tmp_outs/'+args.tsns_output+f'-it{it:03d}', mode='wb') as tsnsfile:
            pickle.dump(trg_senses, tsnsfile)
        
        ### update synset embeddings (10)
        time10 = time.time()
        if args.gd:
            ### TODO use sgd_model
            ### TODO probably handle sizes and/or threshold sparse matrix
            ### TODO see if it's better to implement vstack over cupy alone, from:
            ### https://github.com/scipy/scipy/blob/v1.2.1/scipy/sparse/construct.py#L468-L499
            for i in range(args.gd_emb_steps):
                all_senses = get_sparse_module(vstack((src_senses.get(), trg_senses.get()), format='csr'))
                cc_grad = all_senses.T.dot(xp.concatenate((xw[:src_size], zw[:trg_size])) - all_senses.dot(cc))
                ### TODO maybe switch to norm-based clipping (needs nan handling)
                #cc_grad /= (args.gd_clip * xp.linalg.norm(cc_grad,axis=1))[0]
                cc_grad.clip(-args.gd_clip, args.gd_clip, out=cc_grad)
                cc += args.gd_lr * cc_grad
        
        else:
            all_senses = get_sparse_module(vstack((src_senses, trg_senses), format='csr'))
            xzecc = xp.concatenate((xw[:src_size], zw[:trg_size])).T\
                        .dot(all_senses.toarray()).T  # sense_size * emb_dim
            all_sns_psinv = psinv(all_senses.get(), dtype, args.inv_delta)  ### TODO only update target side? We still have src_sns_psinv [it doesn't matter, dimensions are the same]
            cc[:] = all_sns_psinv.dot(xzecc)
            
        if args.verbose:
            print(f'synset embedding update: {time.time()-time10:.2f}', file=sys.stderr)
            objective = ((xp.linalg.norm(xw[:src_size] - get_sparse_module(src_senses).dot(cc),'fro')) ** 2\
                            + (xp.linalg.norm(zw[:trg_size] - get_sparse_module(trg_senses).dot(cc),'fro')) ** 2) / 2 \
                        + args.reglamb * trg_senses.sum()  # TODO consider thresholding reg part
            objective = float(objective)
            print(f'objective: {objective:.3f}')
        
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
            
            zecc.fill(0.)
            for i in range(0, trg_size, args.trg_batch):
                end_idx = min(i+args.trg_batch, trg_size)
                zecc += z[i:end_idx].T.dot(get_sparse_module(trg_senses[i:end_idx]).toarray()).T
            # this used to work instead of above, until 552f02e
            #zecc = z[:trg_size].T.dot(get_sparse_module(trg_senses).toarray()).T
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
            trg_senses_l1 = float(trg_senses.sum())
            src_obj = float(xp.linalg.norm(xw[:src_size] - get_sparse_module(src_senses).dot(cc),'fro')) ** 2
            trg_obj = float(xp.linalg.norm(zw[:trg_size] - get_sparse_module(trg_senses).dot(cc),'fro')) ** 2
            objective = ((src_obj + trg_obj) / 2) + args.reglamb * trg_senses_l1  # TODO consider thresholding reg part
            if args.verbose:
                print(f'objective calculation: {time.time()-time_obj:.2f}', file=sys.stderr)
            ### TODO create bilingual dictionary?
            
            if objective - best_objective <= -args.threshold:
                last_improvement = it
                best_objective = objective

            # Logging
            duration = time.time() - t
            if args.verbose:
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('objective: {0:.3f}'.format(objective), file=sys.stderr)
                print('target senses l_1 norm: {0:.3f}'.format(trg_senses_l1), file=sys.stderr)
                print('drop probability: {0:.1f}%'.format(100 - 100*keep_prob), file=sys.stderr)
                print(file=sys.stderr)
                sys.stderr.flush()
            if args.log is not None:
                print(f'{it}\t{objective:.3f}\t{src_obj:.3f}\t{trg_obj:.3f}\t{trg_senses_l1:.3f}\t{duration:.6f}\t{trg_senses.getnnz()}', file=log)
                log.flush()

        t = time.time()
        it += 1

    # Write mapped embeddings
    with open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape') as srcfile:
        embeddings.write(src_words, xw, srcfile)
    with open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape') as trgfile:
        embeddings.write(trg_words, zw, trgfile)
    
    # Write target sense mapping
    with open(args.tsns_output, mode='wb') as tsnsfile:
        pickle.dump(trg_senses, tsnsfile)

if __name__ == '__main__':
    main()
