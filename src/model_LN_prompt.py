import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts


def freeze_clip_except_visual_layernorms(model):
    model.requires_grad_(False)
    for module in model.visual.modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

class Model(pl.LightningModule):
    def __init__(self, class_names):
        super().__init__()
        self.save_hyperparameters({'class_names': list(class_names)})

        self.opts = opts
        self.class_names = sorted(class_names)
        self.category_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        freeze_clip_except_visual_layernorms(self.clip)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)
        self.cls_loss_weight = self.opts.cls_loss_weight

        self.best_metric = -1e3
        self.validation_step_outputs = []
        self.register_buffer('class_text_features', self._build_class_text_features())

    def configure_optimizers(self):
        visual_ln_params = [param for param in self.clip.visual.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam([
            {'params': visual_ln_params, 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def _build_class_text_features(self):
        prompts = [self._category_prompt(name) for name in self.class_names]
        text_tokens = clip.tokenize(prompts)
        with torch.no_grad():
            text_features = self.clip.encode_text(text_tokens)
        return F.normalize(text_features.float(), dim=-1)

    @staticmethod
    def _category_prompt(category_name):
        readable_name = category_name.replace('_', ' ')
        return 'a photo of a {}'.format(readable_name)

    def _category_targets(self, categories):
        return torch.tensor(
            [self.category_to_idx[category] for category in categories],
            device=self.device,
            dtype=torch.long
        )

    def _classification_logits(self, feat):
        image_features = F.normalize(feat.float(), dim=-1)
        text_features = F.normalize(self.class_text_features.float(), dim=-1)
        logit_scale = self.clip.logit_scale.exp().float()
        return logit_scale * image_features @ text_features.t()

    def _classification_loss(self, feat, categories):
        logits = self._classification_logits(feat)
        targets = self._category_targets(categories)
        return F.cross_entropy(logits, targets)

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        sketch_cls_loss = self._classification_loss(sk_feat, category)
        photo_cls_loss = self._classification_loss(img_feat, category)
        cls_loss = sketch_cls_loss + photo_cls_loss
        loss = triplet_loss + self.cls_loss_weight * cls_loss

        self.log('train_triplet_loss', triplet_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_sketch_cls_loss', sketch_cls_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_photo_cls_loss', photo_cls_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss)
        self.validation_step_outputs.append((
            sk_feat.detach().cpu(),
            img_feat.detach().cpu(),
            list(category)
        ))
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    @staticmethod
    def _retrieval_metrics_at_k(scores, target, top_k):
        top_k = min(top_k, scores.numel())
        if top_k == 0:
            zero = scores.new_tensor(0.0)
            return zero, zero

        top_indices = torch.argsort(scores, descending=True)[:top_k]
        retrieved = target[top_indices].float()
        precision_at_k = retrieved.mean()

        total_relevant = int(target.sum().item())
        if total_relevant == 0:
            return scores.new_tensor(0.0), precision_at_k

        positions = torch.arange(1, top_k + 1, dtype=torch.float32)
        precision_curve = torch.cumsum(retrieved, dim=0) / positions
        ap_at_k = (precision_curve * retrieved).sum() / min(total_relevant, top_k)
        return ap_at_k, precision_at_k

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        Len = len(outputs)
        if Len == 0:
            return
        query_feat_all = torch.cat([outputs[i][0] for i in range(Len)])
        gallery_feat_all = torch.cat([outputs[i][1] for i in range(Len)])
        all_category = np.array(sum([outputs[i][2] for i in range(Len)], []))


        ## Category-level SBIR Metrics at k=200
        gallery = gallery_feat_all
        ap_at_200 = torch.zeros(len(query_feat_all), dtype=torch.float32)
        precision_at_200 = torch.zeros(len(query_feat_all), dtype=torch.float32)
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            scores = -1 * self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_category == category)] = True
            ap_at_200[idx], precision_at_200[idx] = self._retrieval_metrics_at_k(scores, target, top_k=200)

        mAP_200 = torch.mean(ap_at_200)
        P_200 = torch.mean(precision_at_200)
        self.log('mAP_200', mAP_200, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('P_200', P_200, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > mAP_200.item()) else mAP_200.item()
        if self.trainer.is_global_zero:
            print('epoch {}: mAP@200={:.4f}, P@200={:.4f}, best mAP@200={:.4f}'.format(
                self.current_epoch + 1,
                mAP_200.item(),
                P_200.item(),
                self.best_metric))
        self.validation_step_outputs.clear()
