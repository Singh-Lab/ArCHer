from retriever import *
import argparse
import json

# One training run, configured by flags; checkpoints go to ARCHER_ARTIFACT_PATH (retriever.artifact_path()) as
# {prefix}_ckpt_ep*.pt plus a resumable {prefix}_state.pt. Exit code 0 = all epochs done, 3 = stopped at --time_budget
# (state saved; rerun the same command to resume), so a slurm chain can resubmit on 3. The run's flags (without
# --time_budget) are saved as {prefix}_args.json, where eval_ckpts.py takes the backbone and scorer from.
#   PYTHONPATH=. python experiments/train.py --prefix pfam_dtw_geo --data pfam --scorer dtw --sim geometric
# --scorer name -> sim.py class; 'qnw' is qNWSim (formerly 'nwpg', then 'gap_dtw': runs and result files were renamed
# 2026-10-29); 'qnw_test' is qNWTestSim (qNW on the mutual softmax's picks: tanh match, pick gaps); 'qnw_gamma' is
# qNWGammaSim (qNW with a learned match bias gamma); 'qnw_mirror' is qNWMirrorSim (match and gaps on the same
# mutual-softmax picks, both in mismatch units); 'qnw_mirror_gamma' is qNWMirrorGammaSim (qNW-mirror, gamma learned);
# 'qnw_null' is qNWNullSim (qNW-mirror with the gap's flank probability normalized against a learned null, gamma 1)
SCORERS = {'dense': DenseSim, 'max_sim': MaxSim, 'dtw': DTWSim, 'query_dtw': QueryDTWSim, 'soft_dtw': SoftDTWSim,
           'qnw': qNWSim, 'qnw_test': qNWTestSim, 'qnw_gamma': qNWGammaSim, 'qnw_mirror': qNWMirrorSim, 'qnw_mirror_gamma': qNWMirrorGammaSim,
           'qnw_null': qNWNullSim}
QNW_SCORERS = ('qnw', 'qnw_test', 'qnw_gamma', 'qnw_mirror', 'qnw_mirror_gamma', 'qnw_null')                   # take the gap flags below
# --gap_rel geo's gap init, set so qNW-geogap starts with qNW's gap costs: GeoGap's flank S and the cosine flank matched
# at their median and 99th percentile (untrained ProtT5, Pfam train positives); ext_alpha = alpha - 1.4, as qNW's affine
GEO_GAP_INIT = dict(alpha=-5.685, beta=19.286, ext_alpha=-7.085, ext_beta=19.286)   # S median -0.299, p99 -0.201: a sharp switch


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True)                     # checkpoint name prefix
    ap.add_argument('--data', choices=('pfam',) + SCOPE_SUBSETS, default='pfam')   # trains on its --train_split split
    ap.add_argument('--train_split', choices=('train', 'dev'), default='train')   # pfam: dev, the held-out sequences of train's families
    ap.add_argument('--backbone', default='esm2-35M')
    ap.add_argument('--unfreeze', type=int, default=3)              # last n PLM layers trained
    ap.add_argument('--scorer', choices=SCORERS, default='dtw')
    ap.add_argument('--sim', choices=('cosine', 'arithmetic', 'geometric'), default='cosine')   # residue similarity
    ap.add_argument('--sim_temp', type=float, default=0.5)          # AffineMutualSoftmax temperature
    ap.add_argument('--beta', type=float, default=0.)               # AffineMutualSoftmax beta
    ap.add_argument('--gaps', choices=('linear', 'affine'), default='linear')   # qnw: linear or affine (Gotoh) gaps
    ap.add_argument('--gap_form', choices=('softplus', 'hinge'), default='softplus')   # qnw: G's parameterization (sim.qNWSim)
    ap.add_argument('--gap_beta_max', type=float, default=None)    # qnw hinge: cap on the switch's sharpness beta; None = uncapped
    ap.add_argument('--gap_rel', choices=('cosine', 'geo'), default='cosine')   # qnw: the flank similarity phi of the gaps
                                                                    # (sim.CosineGap, or sim.GeoGap: the match's own S)
    ap.add_argument('--cls_alpha', type=float, default=0.)          # > 0: ClsEncoder + ClsAugmentedInfoNCE
    ap.add_argument('--distill_from', default=None)                 # a trained run's prefix: its checkpoint, frozen, is the teacher of
                                                                    # infonce.DistillInfoNCE (retriever.DistillTrainer)
    ap.add_argument('--distill_epoch', type=int, default=9)         # the teacher's checkpoint
    ap.add_argument('--distill_weight', type=float, default=1.)     # weight of the KL term
    ap.add_argument('--nce_weight', type=float, default=1.)         # weight of InfoNCE next to it; 0: distillation only
    ap.add_argument('--teacher_temperature', type=float, default=0.05)   # softmax temperature of the teacher's scores
    ap.add_argument('--init_from', default=None)                    # a trained run's prefix: start from its checkpoint (same
                                                                    # backbone and scorer), not from the PLM; its "untrained" is the PLM
    ap.add_argument('--init_epoch', type=int, default=9)            # that run's checkpoint
    ap.add_argument('--temperature', type=float, default=0.05)      # InfoNCE temperature
    ap.add_argument('--max_len', type=int, default=1022)            # drop longer training sequences; 0 keeps all
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--length_buckets', type=int, default=0)        # > 0: length-sorted batches (RetrieverTrainer.epoch_batches),
                                                                    # sorted within chunks of this many batches
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--save_every', type=int, default=1)
    ap.add_argument('--lr_backbone', type=float, default=1e-5)
    ap.add_argument('--lr_head', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--time_budget', type=float, default=None)      # seconds
    ap.add_argument('--seed', type=int, default=2603)
    return ap


def residue_sim(args):
    if args.sim == 'cosine':
        return COSINE
    return AffineMutualSoftmax(args.sim_temp, average=args.sim, beta=args.beta)


def scorer(args):
    # args: train.py's flags, or eval_ckpts.run_config's (a run's saved _args.json). Older runs may have been trained
    # with options since removed (--gamma, a learned --gap_rel): those cannot be rebuilt, so they are refused, not
    # silently scored with the defaults
    assert getattr(args, 'gamma', None) is None, f'--gamma {args.gamma}: removed option, run not reproducible here'
    gap_rel = getattr(args, 'gap_rel', None) or 'cosine'
    assert gap_rel in ('cosine', 'geo'), f'--gap_rel {gap_rel}: removed option'
    assert gap_rel == 'cosine' or args.scorer == 'qnw', '--gap_rel geo: qnw only (qnw_test has its own gaps, qnw_gamma an S that moves with gamma)'
    kw = {'sim': residue_sim(args)}
    if args.scorer in QNW_SCORERS:
        # runs from before --gaps / --gap_form: linear gaps, softplus form
        kw['gaps'] = getattr(args, 'gaps', None) or 'linear'
        kw['form'] = getattr(args, 'gap_form', None) or 'softplus'
        kw['beta_max'] = getattr(args, 'gap_beta_max', None)
    if gap_rel == 'geo':
        kw.update(relation='geo', **GEO_GAP_INIT)
    return SCORERS[args.scorer](**kw)


def attach_scorer(encoder, sim_fn):
    # a scorer with learned parameters (qNWSim's gap costs and gap relation) becomes encoder.scorer: they then train
    # with the head parameters (RetrieverTrainer.param_groups) and are saved in, and loaded from, the encoder's
    # checkpoints; bind first, so a learned gap relation creates its parameters at the embedding size
    if isinstance(sim_fn, nn.Module):
        if hasattr(sim_fn, 'bind'):
            sim_fn.bind(encoder.plm_wrapper.hidden_size)
        encoder.scorer = sim_fn.to('cuda')


def dataset(args):
    kw = {'max_len': args.max_len or None}
    if args.data == 'pfam':
        return Pfam(getattr(args, 'train_split', 'train'), label='clan', **kw)
    assert getattr(args, 'train_split', 'train') == 'train', '--train_split dev: pfam only'
    return SCOPe(args.data, 'train', **kw)


def teacher(args):
    # the frozen teacher of --distill_from: its encoder at --distill_epoch, with its scorer (and learned scorer parameters)
    # attached, on the student's backbone (cached or not), which must be the same PLM the teacher was trained on
    path = os.path.join(artifact_path(), f'{args.distill_from}_args.json')
    cfg = vars(parser().parse_args(['--prefix', args.distill_from]))
    cfg.update(json.load(open(path)))
    cfg = argparse.Namespace(**cfg)
    base = lambda b: b[:-len('-cached')] if b.endswith('-cached') else b
    assert base(cfg.backbone) == base(args.backbone), f'teacher trained on {cfg.backbone}, student on {args.backbone}'
    enc = Encoder(backbone=args.backbone, unfreeze_last_n=cfg.unfreeze).to('cuda')
    attach_scorer(enc, scorer(cfg))
    ckpt = os.path.join(artifact_path(), f'{args.distill_from}_ckpt_ep{args.distill_epoch:03d}.pt')
    enc.load_checkpoint(torch.load(ckpt, map_location='cuda'))
    print(f'teacher {args.distill_from} epoch {args.distill_epoch}: {enc.scorer!r}')
    return enc


if __name__ == '__main__':
    args = parser().parse_args()
    print(vars(args))
    os.makedirs(artifact_path(), exist_ok=True)
    with open(os.path.join(artifact_path(), f'{args.prefix}_args.json'), 'w') as f:
        json.dump({k: v for k, v in vars(args).items() if k != 'time_budget'}, f, indent=1)
    set_seed(args.seed)
    kw = {'backbone': args.backbone, 'unfreeze_last_n': args.unfreeze}
    assert not (args.distill_from and args.cls_alpha > 0), '--distill_from with --cls_alpha: not supported'
    assert args.distill_from or args.nce_weight == 1., '--nce_weight: with --distill_from only'
    encoder = (ClsEncoder if args.cls_alpha > 0 else Encoder)(**kw).to('cuda')
    if args.distill_from:
        loss_fn = DistillInfoNCE(scorer(args), temperature=args.temperature, distill_weight=args.distill_weight,
                                 nce_weight=args.nce_weight, teacher_temperature=args.teacher_temperature)
    elif args.cls_alpha > 0:
        loss_fn = ClsAugmentedInfoNCE(scorer(args), alpha=args.cls_alpha, temperature=args.temperature)
    else:
        loss_fn = InfoNCE(scorer(args), temperature=args.temperature)
    attach_scorer(encoder, loss_fn.scorer)
    if args.init_from:
        # before fit(): a resumed run's _state.pt then overrides it
        init = os.path.join(artifact_path(), f'{args.init_from}_ckpt_ep{args.init_epoch:03d}.pt')
        encoder.load_checkpoint(torch.load(init, map_location='cuda'))
        print(f'init from {args.init_from} epoch {args.init_epoch}')
    print(loss_fn)
    data = dataset(args)
    kw = {'batch_size': args.batch_size, 'ckpt_prefix': args.prefix}
    trainer = (DistillTrainer(data, encoder, loss_fn, teacher(args), **kw) if args.distill_from
               else RetrieverTrainer(data, encoder, loss_fn, **kw))
    done = trainer.fit(num_epochs=args.epochs, save_every=args.save_every, resume=True, time_budget=args.time_budget,
                       seed=args.seed, length_buckets=args.length_buckets,
                       lr_backbone=args.lr_backbone, lr_head=args.lr_head, wd=args.wd)
    raise SystemExit(0 if done else 3)
