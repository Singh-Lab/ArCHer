from utils import *
from plm_wrapper import *


class Encoder(nn.Module):
    """
    Sequences -> padded per-residue embeddings (B, L, D) and a bool mask (B, L), from a PLM (backbone, a name from
    PLMWrapper.from_name) whose last unfreeze_last_n layers train. Subclasses add a trainable head on top
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.plm_wrapper = PLMWrapper.from_name(
            kwargs.get('backbone', 'esm2-35M'),
            unfreeze_last_n=kwargs.get('unfreeze_last_n', 3)
        )
        # frozen parameters a checkpoint load moved off their initial value, so trainable_state_dict keeps them
        self._changed_frozen = set()

    def frozen_keys(self):
        # remove_duplicate=False also lists tied aliases (ESM's lm_head.decoder.weight is the input embedding matrix)
        return {n for n, p in self.named_parameters(remove_duplicate=False) if not p.requires_grad}

    def trainable_state_dict(self):
        # the state_dict minus frozen parameters still at their initial value (the pretrained backbone): those come
        # back as is when the model is rebuilt, so a checkpoint only needs the rest (trainable weights, buffers, and any
        # frozen weight a load changed)
        skip = self.frozen_keys() - self._changed_frozen
        return {k: v for k, v in self.state_dict().items() if k not in skip}

    def load_checkpoint(self, state):
        # a trainable_state_dict or a full state_dict; whatever it lacks must be a frozen parameter or a buffer (both
        # kept at their initial value); a missing trainable parameter is an error
        own, frozen = self.state_dict(), self.frozen_keys()
        changed = {k for k, v in state.items() if k in frozen and not torch.equal(own[k], v.to(own[k].device))}
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad = [k for k in missing if k not in frozen and k not in dict(self.named_buffers())]
        assert not unexpected and not bad, f'checkpoint does not fit the model: unexpected {unexpected[:5]}, missing trainable {bad[:5]}'
        self._changed_frozen |= changed
        return self

    def head(self, emb, mask):
        # trainable layers on the padded PLM embeddings; none here
        return emb

    def embed(self, seqs):
        embeddings, _ = self.plm_wrapper(seqs)
        emb, mask = pad_embeddings(embeddings)
        return self.head(emb, mask), mask

    def forward(self, q_seqs, d_seqs=None):
        # q_seqs only: (emb, mask); with d_seqs: (q_emb, q_mask, d_emb, d_mask), both sides in one PLM call
        if d_seqs is None:
            return self.embed(q_seqs)
        embeddings, _ = self.plm_wrapper(q_seqs + d_seqs)
        q_emb, q_mask = pad_embeddings(embeddings[:len(q_seqs)])
        d_emb, d_mask = pad_embeddings(embeddings[len(q_seqs):])
        return self.head(q_emb, q_mask), q_mask, self.head(d_emb, d_mask), d_mask


class ClsEncoder(Encoder):
    """
    Encoder that also returns each sequence's CLS embedding (the final hidden state at ESM's <cls>, or at ProtT5's </s>,
    which stands in for one; from the same PLM pass, the mean over its tiles for a sequence longer than max_len), for losses that train on it next to the per-residue
    embeddings (infonce.ClsAugmentedInfoNCE). encoder(q, d) -> (q_emb, q_mask, d_emb, d_mask, q_cls, d_cls);
    encoder(seqs) -> (emb, mask) as Encoder, so RetrieverEval scores it unchanged. No parameters of its own, so its
    checkpoints are Encoder's
    """
    def embed_cls(self, seqs):
        # (emb, mask, cls): the padded residue embeddings and the (B, D) CLS embeddings
        embeddings, _, cls = self.plm_wrapper(seqs, return_cls=True)
        emb, mask = pad_embeddings(embeddings)
        return self.head(emb, mask), mask, cls

    def forward(self, q_seqs, d_seqs=None):
        if d_seqs is None:
            return self.embed(q_seqs)
        embeddings, _, cls = self.plm_wrapper(q_seqs + d_seqs, return_cls=True)
        nq = len(q_seqs)
        q_emb, q_mask = pad_embeddings(embeddings[:nq])
        d_emb, d_mask = pad_embeddings(embeddings[nq:])
        return self.head(q_emb, q_mask), q_mask, self.head(d_emb, d_mask), d_mask, cls[:nq], cls[nq:]


