import os

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from msbert.utils import set_seed, maxsum, cls_score, mv_score
from msbert.config import TrainConfig
from msbert.dataloader import DataLoader
from msbert.modeling import MSBert


def score(qry_repr: dict, doc_repr: dict, pairwise: bool) -> torch.Tensor:
    cls_scores = cls_score(qry_repr["cls"], doc_repr["cls"], pairwise)
    tok_scores = maxsum(mv_score(qry_repr["tok"]["repr"], doc_repr["tok"]["repr"], pairwise))
    span_scores = maxsum(mv_score(qry_repr["span"]["repr"], doc_repr["span"]["repr"], pairwise))

    return cls_scores + tok_scores + span_scores

def info_nce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(scores, labels, reduction="mean")

def run(config: TrainConfig, dataset_dir: str, log_dir: str, checkpoint_dir: str) -> None:
    set_seed(42)

    # load training data
    dataloader = DataLoader(
        config,
        **{
            item: os.path.join(dataset_dir, f"{item}.jsonl")
            for item in ("records", "queries", "documents")
        }
    )

    # load model
    model = MSBert(config.model_config)
    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    scheduler = None
    if config.warmup is not None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup,
            num_training_steps=config.maxsteps
        )

    writer = SummaryWriter(log_dir)

    # the constants required for the loss function
    nway = 1 + config.n_negative
    temperature = torch.full((nway,), config.temperature, requires_grad=False, device=config.device)

    for batch_idx, batch in enumerate(dataloader):
        queries, documents = batch

        # forward pass
        Q = model.encode(*queries)
        D = model.encode(*documents)

        # calculate Query - Document score
        scores = score(Q, D, config.neg_sampling is False)

        if config.neg_sampling:
            labels = torch.arange(0, scores.shape[1], nway, dtype=torch.long, device=config.device)
        else:
            labels = torch.zeros(config.bsize, dtype=torch.long, device=config.device)
        loss = info_nce_loss(scores / temperature, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        # update parameters and lr
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("loss", loss.item(), batch_idx)

        if batch_idx >= config.maxsteps:
            break

        # free memory
        if batch_idx % 500 == 0:
            torch.cuda.empty_cache()
    model.save_pretrained(checkpoint_dir)