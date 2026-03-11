import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts



def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class Model(pl.LightningModule):
    def __init__(self, class_names=None):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)
        self.cls_loss_weight = self.opts.cls_loss_weight
        self.class_names = sorted(class_names) if class_names is not None else []

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)
        self.patch_shuffle_loss_weight = self.opts.patch_shuffle_loss_weight
        self.patch_shuffle_grid = self.opts.patch_shuffle_grid
        self.patch_shuffle_loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=self.opts.patch_shuffle_margin)

        self.best_metric = -1.0
        self.validation_step_outputs = []
        if self.cls_loss_weight > 0:
            self.category_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.register_buffer('class_text_features', self._build_class_text_features(), persistent=False)
        else:
            self.category_to_idx = {}
            self.class_text_features = None

    def configure_optimizers(self):
        clip_trainable_params = [param for param in self.clip.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam([
            {'params': clip_trainable_params, 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def _build_class_text_features(self):
        prompts = ['a photo of a {}'.format(name.replace('_', ' ')) for name in self.class_names]
        text_tokens = clip.tokenize(prompts)
        with torch.no_grad():
            text_features = self.clip.encode_text(text_tokens)
        return F.normalize(text_features.float(), dim=-1)

    def _classification_loss(self, feat, categories):
        image_features = F.normalize(feat.float(), dim=-1)
        text_features = self.class_text_features.float()
        logit_scale = self.clip.logit_scale.exp().float()
        logits = logit_scale * image_features @ text_features.t()
        targets = torch.tensor(
            [self.category_to_idx[category] for category in categories],
            device=logits.device,
            dtype=torch.long
        )
        return F.cross_entropy(logits, targets)

    def _random_permutations(self, batch_size, device):
        num_patches = self.patch_shuffle_grid * self.patch_shuffle_grid
        perm_1 = torch.stack([
            torch.randperm(num_patches, device=device) for _ in range(batch_size)
        ])
        perm_2 = torch.stack([
            torch.randperm(num_patches, device=device) for _ in range(batch_size)
        ])

        if num_patches > 1:
            same_perm_mask = torch.all(perm_1 == perm_2, dim=1)
            if same_perm_mask.any():
                perm_2[same_perm_mask] = torch.roll(perm_1[same_perm_mask], shifts=1, dims=1)

        return perm_1, perm_2

    def _shuffle_patches(self, images, permutations):
        batch_size, channels, height, width = images.shape
        grid = self.patch_shuffle_grid
        if height % grid != 0 or width % grid != 0:
            raise ValueError('image size must be divisible by patch_shuffle_grid')

        patch_h = height // grid
        patch_w = width // grid
        num_patches = grid * grid

        patches = images.view(batch_size, channels, grid, patch_h, grid, patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, num_patches, channels, patch_h, patch_w)

        gather_index = permutations.view(batch_size, num_patches, 1, 1, 1)
        gather_index = gather_index.expand(-1, -1, channels, patch_h, patch_w)
        shuffled_patches = torch.gather(patches, 1, gather_index)

        shuffled_images = shuffled_patches.view(batch_size, grid, grid, channels, patch_h, patch_w)
        shuffled_images = shuffled_images.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
        return shuffled_images

    def _patch_shuffle_loss(self, sk_tensor, img_tensor):
        perm_1, perm_2 = self._random_permutations(sk_tensor.shape[0], sk_tensor.device)

        shuffled_sketch = self._shuffle_patches(sk_tensor, perm_1)
        shuffled_photo_same = self._shuffle_patches(img_tensor, perm_1)
        shuffled_photo_diff = self._shuffle_patches(img_tensor, perm_2)

        shuffled_sketch_feat = self.forward(shuffled_sketch, dtype='sketch')
        shuffled_photo_same_feat = self.forward(shuffled_photo_same, dtype='image')
        shuffled_photo_diff_feat = self.forward(shuffled_photo_diff, dtype='image')
        return self.patch_shuffle_loss_fn(
            shuffled_sketch_feat,
            shuffled_photo_same_feat,
            shuffled_photo_diff_feat
        )

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category, instance_id = batch
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        loss = triplet_loss
        if self.cls_loss_weight > 0:
            sketch_cls_loss = self._classification_loss(sk_feat, category)
            photo_cls_loss = self._classification_loss(img_feat, category)
            loss = loss + self.cls_loss_weight * (sketch_cls_loss + photo_cls_loss)
            self.log('train_sketch_cls_loss', sketch_cls_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train_photo_cls_loss', photo_cls_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if self.patch_shuffle_loss_weight > 0:
            patch_shuffle_loss = self._patch_shuffle_loss(sk_tensor, img_tensor)
            loss = loss + self.patch_shuffle_loss_weight * patch_shuffle_loss
            self.log('train_patch_shuffle_loss', patch_shuffle_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log('train_triplet_loss', triplet_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category, instance_id = batch
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.validation_step_outputs.append((
            sk_feat.detach().cpu(),
            img_feat.detach().cpu(),
            list(category),
            list(instance_id)
        ))
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return

        query_feat_all = torch.cat([item[0] for item in outputs])
        gallery_feat_all = torch.cat([item[1] for item in outputs])
        all_category = np.array(sum([item[2] for item in outputs], []))
        all_instance_id = np.array(sum([item[3] for item in outputs], []))

        gallery_by_key = {}
        for feat, category, instance_id in zip(gallery_feat_all, all_category, all_instance_id):
            key = (category, instance_id)
            if key not in gallery_by_key:
                gallery_by_key[key] = feat

        gallery_keys = list(gallery_by_key.keys())
        gallery_feats = torch.stack([gallery_by_key[key] for key in gallery_keys])

        correct_at_1 = 0
        correct_at_5 = 0
        for query_feat, category, target_instance_id in zip(query_feat_all, all_category, all_instance_id):
            category_indices = [
                idx for idx, (gallery_category, _) in enumerate(gallery_keys)
                if gallery_category == category
            ]
            if not category_indices:
                continue

            category_gallery = gallery_feats[category_indices]
            category_gallery_ids = [gallery_keys[idx][1] for idx in category_indices]
            scores = -1 * self.distance_fn(query_feat.unsqueeze(0), category_gallery)
            ranked_indices = torch.argsort(scores, descending=True)

            if category_gallery_ids[ranked_indices[0].item()] == target_instance_id:
                correct_at_1 += 1

            top_k = min(5, len(category_gallery_ids))
            top_ids = [category_gallery_ids[idx] for idx in ranked_indices[:top_k].tolist()]
            if target_instance_id in top_ids:
                correct_at_5 += 1

        num_queries = max(1, len(query_feat_all))
        acc_1 = 100.0 * correct_at_1 / num_queries
        acc_5 = 100.0 * correct_at_5 / num_queries

        self.log('acc_1', acc_1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('acc_5', acc_5, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > acc_1) else acc_1
        if self.trainer.is_global_zero:
            print('epoch {}: Acc@1={:.2f}, Acc@5={:.2f}, best Acc@1={:.2f}'.format(
                self.current_epoch + 1,
                acc_1,
                acc_5,
                self.best_metric))
        self.validation_step_outputs.clear()
