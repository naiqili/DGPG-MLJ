from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # using specific GPU
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from compatible.likelihoods import MultiClass, Gaussian
from compatible.kernels import RBF, White
from gpflow.models.svgp import SVGP
from gpflow.training import AdamOptimizer, ScipyOptimizer
from scipy.stats import mode
from scipy.cluster.vq import kmeans2
import gpflow
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow import autoflow, params_as_tensors, ParamList
import pandas as pd
import itertools
pd.options.display.max_rows = 999
import gpflow_monitor

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.io import loadmat
from gpflow_monitor import *
print('tf_ver:', tf.__version__, 'gpflow_ver:', gpflow.__version__)
from tensorflow.python.client import device_lib
print('avail devices:\n'+'\n'.join([x.name for x in device_lib.list_local_devices()]))
from jack_utils.common import time_it
import sys
import gpflow.training.monitor as mon

# our impl
from dgp_graph import *


import argparse

parser = argparse.ArgumentParser(description='main.')
parser.add_argument('--infile', type=str)
parser.add_argument('--outfile', type=str)
parser.add_argument('--sizen', type=int)
parser.add_argument('--sizem', type=int)
parser.add_argument('--gene', type=int)
parser.add_argument('--iter', type=int, default=20000)
parser.add_argument('--ktype', type=str, default='Poly1')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--delta', type=float, default=-1)

args = parser.parse_args()

np.random.seed(123456)

cc=0.1
loc=60
tts=200
# nodes = 100
# M = 40
sizen = args.sizen
M = args.sizem
nodes = args.gene
inc=False
maxiter=args.iter
lr=args.lr

fname = args.infile
raw_data = []
gene_name = []

with open(fname) as f:
    gene_name = f.readline().strip().split()
    for l in f:
        row = [float(x) for x in l.strip().split()]
        if len(row) > 0:
            raw_data.append(row)
            sizen -= 1
            if sizen==0:
                break
            
data = np.asarray(raw_data)

trX0 = data
trY0 = data

trX1, trY1 = trX0[:,:,None], trY0[:,:,None]

def normalize_data(data, mu, std):
    res = (data-mu) / std
    return res

def unnormalize_data(data, mu, std):
    res = data * std.reshape(1,-1) + mu.reshape(1,-1)
    return res

adj = np.ones((nodes,nodes)) - np.eye(nodes)

mu_trX0, std_trX0 = np.mean(trX1, axis=0, keepdims=True), np.std(trX1, axis=0, keepdims=True)
mu_trY0, std_trY0 = np.mean(trY1, axis=0, keepdims=True), np.std(trY1, axis=0, keepdims=True)

trX = normalize_data(trX1, mu_trX0, std_trX0)
trYY = trY = normalize_data(trY1, mu_trY0, std_trY0)

Z = np.stack([kmeans2(trX[:,i]+np.random.randn(*trX[:,i].shape)*0.01, M, minit='points')[0] for i in range(nodes)],axis=1)  # (M=s2=10, n, d_in=5)
print('inducing points Z: {}'.format(Z.shape))

adj = adj.astype('float64')
input_adj = adj # adj  / np.identity(adj.shape[0]) /  np.ones_like(adj)

time_vec=np.ones(trX.shape[0], )

with gpflow.defer_build():
    m_dgpg = DGPG(trX, trYY, Z, time_vec, [1], Gaussian(), input_adj,
                  agg_op_name='concat3d', ARD=True,
                  is_Z_forward=True, mean_trainable=False, out_mf0=True,
                  num_samples=1, minibatch_size=1,
                  #kern_type='Matern32', 
                  #kern_type='RBF', 
                  kern_type=args.ktype, 
                  wfunc='logi'
                 )
    # m_sgp = SVGP(X, Y, kernels, Gaussian(), Z=Z, minibatch_size=minibatch_size, whiten=False)
m_dgpg.compile()
model = m_dgpg

session = m_dgpg.enquire_session()
optimiser = gpflow.train.AdamOptimizer(lr)
# optimiser = gpflow.train.ScipyOptimizer()
global_step = mon.create_global_step(session)

def rmse(v1, v2):
    return np.sqrt(np.mean((v1.reshape(-1)-v2.reshape(-1))**2))

model.X.update_cur_n(trX.shape[0]-1,cc=cc,loc=loc)
model.Y.update_cur_n(trX.shape[0]-1,cc=cc,loc=loc)

pred_res = []

exp_path="./exp/tmp-cc%d" % int(cc)
#exp_path="./exp/temp"

print_task = mon.PrintTimingsTask()\
    .with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(10))\

checkpoint_task = mon.CheckpointTask(checkpoint_dir=exp_path)\
        .with_name('checkpoint')\
        .with_condition(mon.PeriodicIterationCondition(15))\

with mon.LogdirWriter(exp_path) as writer:
    tensorboard_task = mon.ModelToTensorBoardTask(writer, model)\
        .with_name('tensorboard')\
        .with_condition(mon.PeriodicIterationCondition(100))\
        .with_exit_condition(True)
    monitor_tasks = [tensorboard_task] # [print_task, tensorboard_task]

    with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        optimiser.minimize(model, step_callback=monitor, global_step=global_step, maxiter=maxiter)
        #optimiser.minimize(model, step_callback=monitor, maxiter=maxiter)

model.trainable = False

for l in range(1):
    model.layers[l].kern.trainable = True
    
model.likelihood.trainable = True

optimiser = gpflow.train.AdamOptimizer(lr)

exp_path+="-retrain_kern"
#exp_path="./exp/temp"

print_task = mon.PrintTimingsTask()\
    .with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(10))\

checkpoint_task = mon.CheckpointTask(checkpoint_dir=exp_path)\
        .with_name('checkpoint')\
        .with_condition(mon.PeriodicIterationCondition(15))\

with mon.LogdirWriter(exp_path) as writer:
    tensorboard_task = mon.ModelToTensorBoardTask(writer, model)\
        .with_name('tensorboard')\
        .with_condition(mon.PeriodicIterationCondition(100))\
        .with_exit_condition(True)
    monitor_tasks = [tensorboard_task] # [print_task, tensorboard_task]
    
       
    with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        optimiser.minimize(model, step_callback=monitor, global_step=global_step, maxiter=maxiter)

rel = np.zeros((nodes, nodes))

for j in range(nodes):
    for i in range(nodes):
        if i==j:
            continue
        rel[i, j] = model.layers[0].kern.lengthscales.value[j, i]
        
# for j in range(nodes):
#     rel[:,j] /= sum(rel[:,j])        
        
idx = np.argsort(rel.reshape(-1))
        
res = []

#for k in idx:
for k in idx[::-1]:
    res.append((k//nodes, k%nodes, rel[k//nodes, k%nodes]/max(rel.reshape(-1))))
        
res_str = ''
for (i, j, v) in res:
#     if v>1e-6:
    if not i==j:
        if args.delta < 0 and v > args.delta:
            res_str += ('%s\t%s\t%f' % (gene_name[i],gene_name[j],v)) + '\r\n'
        
with open(args.outfile, 'w') as fout:
    fout.write(res_str)
        