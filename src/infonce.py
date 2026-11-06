from sim import *


class InfoNCE(nn.Module):
    """
    In-batch contrastive loss: scorer (a SequenceSim) scores every anchor against every positive, (B, B) / temperature,
    and the cross-entropy targets the diagonal (each anchor's own positive). exclude (B, B bool, diagonal False) masks
    in-batch negatives that are actually positives (same sf as the anchor) out of the softmax
    """
    def __init__(self, scorer, temperature=0.05):
        super().__init__()
        self.scorer = scorer
        self.temperature = temperature

    def contrast(self, scores, exclude=None):
        # (B, B) scores -> cross-entropy toward the diagonal at self.temperature, exclude masked out
        logits = scores / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        if exclude is not None:
            logits = logits.masked_fill(exclude.to(logits.device), NEG_INF)
        return F.cross_entropy(logits, targets)

    def forward(self, q_emb, q_mask, d_emb, d_mask, exclude=None):
        return self.contrast(self.scorer(q_emb, q_mask, d_emb, d_mask), exclude)

    def extra_repr(self):
        return f'scorer={self.scorer!r}, temperature={self.temperature}'


class ClsAugmentedInfoNCE(InfoNCE):
    """
    (1 - alpha) * InfoNCE over the residue scorer (e.g. DTW) + alpha * InfoNCE over the CLS embeddings, whose similarity
    is the cosine rescaled to [0, 1], (cos + 1) / 2, so both terms see scores on a similar scale; both use the same
    temperature, diagonal targets and exclude mask. Takes encoder.ClsEncoder's output:
      loss(q_emb, q_mask, d_emb, d_mask, q_cls, d_cls, exclude)
    """
    def __init__(self, scorer, alpha=0.1, temperature=0.05):
        super().__init__(scorer, temperature)
        self.alpha = alpha

    def forward(self, q_emb, q_mask, d_emb, d_mask, q_cls, d_cls, exclude=None):
        res_scores = self.scorer(q_emb, q_mask, d_emb, d_mask)
        loss = self.contrast(res_scores, exclude)
        if self.alpha > 0:
            cls_scores = (F.cosine_similarity(q_cls.float()[:, None], d_cls.float()[None], dim=-1) + 1) / 2
            cls_loss = self.contrast(cls_scores, exclude)
            loss = (1 - self.alpha) * loss + self.alpha * cls_loss
        return loss

    def extra_repr(self):
        return f'{super().extra_repr()}, alpha={self.alpha}'

class DistillInfoNCE(InfoNCE):
    """
    nce_weight * InfoNCE + distill_weight * KL(p_T || p_S), the second a listwise distillation of a teacher's in-batch
    ranking: for each anchor i, p_T(j | i) = softmax_j(T_ij / teacher_temperature) over the teacher's (B, B) scores of
    the same pairs (row i: anchor i as the query) and p_S(j | i) = softmax_j(S_ij / temperature) over the student's.
    The KL term keeps every in-batch document, the same-sf ones that exclude masks out of InfoNCE included: the
    teacher's grading of those is what it passes on. nce_weight 0: distillation only. Takes the teacher's scores:
      loss(q_emb, q_mask, d_emb, d_mask, exclude, teacher_scores)
    """
    def __init__(self, scorer, temperature=0.05, distill_weight=1., nce_weight=1., teacher_temperature=0.05):
        super().__init__(scorer, temperature)
        self.distill_weight, self.nce_weight, self.teacher_temperature = distill_weight, nce_weight, teacher_temperature
        self.last = {}

    def forward(self, q_emb, q_mask, d_emb, d_mask, exclude=None, teacher_scores=None):
        scores = self.scorer(q_emb, q_mask, d_emb, d_mask).float()
        p_t = F.softmax(teacher_scores.float() / self.teacher_temperature, dim=-1)
        log_p_s = F.log_softmax(scores / self.temperature, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction='batchmean')
        loss = self.distill_weight * kl
        nce = self.contrast(scores, exclude) if self.nce_weight > 0 else None
        if nce is not None:
            loss = loss + self.nce_weight * nce
        diag = torch.arange(len(p_t), device=p_t.device)
        # the last batch's terms, and how peaked the teacher's targets are (its mass on the positive), for log_params
        self.last = {'kl': kl.item(), 'teacher_p_pos': p_t[diag, diag].mean().item(),
                     'teacher_p_max': p_t.max(-1).values.mean().item(), **({'nce': nce.item()} if nce is not None else {})}
        return loss

    def log_params(self):
        return self.last

    def extra_repr(self):
        return (f'{super().extra_repr()}, distill_weight={self.distill_weight}, nce_weight={self.nce_weight}, '
                f'teacher_temperature={self.teacher_temperature}')


if __name__ == '__main__':
    # InfoNCE with symmetric hard DTW over 20 batches of Pfam train, frozen ESM2-35M
    from data import *
    from encoder import *
    set_seed(2603)
    loss_fn = InfoNCE(DTWSim())
    encoder = Encoder(backbone='esm2-35M', unfreeze_last_n=0).to('cuda')
    loader = DataLoader(Pfam('train'), batch_size=16, shuffle=True, drop_last=True)
    losses, t = [], time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, batch in enumerate(loader):
            if i == 20:
                break
            labels = batch['sf']
            exclude = (labels.unsqueeze(0) == labels.unsqueeze(1)).fill_diagonal_(False)
            losses.append(loss_fn(*encoder(batch['anchor'], batch['positive']), exclude).item())
    print(loss_fn)
    print(f'mean loss {np.mean(losses):.3f} over {len(losses)} batches (chance: ln 16 = {np.log(16):.3f}), {time.time() - t:.1f}s')
