from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fate_oia.models.coev_evidence_readout import CoEVEvidenceReadout
from fate_oia.models.coev_observers import CorrespondenceObserver, PredicateObserver, TrafficPrimitiveBuilder, inverse_grid_sample_affine
from fate_oia.models.coev_path_lift import CoEVCoupledPathLift
from fate_oia.models.coev_video_decoder import CoEVVideoDecoder
from fate_oia.models.coev_visual_field import CoEVVisualField
from fate_oia.utils.coev_contracts import CoEVInputs


def _pool_tokens(tokens: Tensor, source_hw: tuple[int, int], target_hw: tuple[int, int] = (16, 28)) -> Tensor:
    b, t, n, d = tokens.shape
    return F.adaptive_avg_pool2d(tokens.view(b * t, *source_hw, d).permute(0, 3, 1, 2), target_hw).permute(0, 2, 3, 1).reshape(b, t, -1, d)


class CoEVOIAModel(nn.Module):
    def __init__(self, pretrained_weights: str, chunk_size: int = 1,
                 reason_bias_verified: bool = False, use_mock_dino: bool = False) -> None:
        super().__init__(); self.chunk_size = chunk_size
        self.visual_field = CoEVVisualField(pretrained_weights, use_mock=use_mock_dino)
        self.predicate_observer = PredicateObserver()
        self.correspondence_observer = CorrespondenceObserver()
        self.primitive_builder = TrafficPrimitiveBuilder()
        self.path_lift = CoEVCoupledPathLift()
        self.video_decoder = CoEVVideoDecoder(reason_bias_verified=reason_bias_verified)
        self.evidence_readout = CoEVEvidenceReadout()
        self.capture_gradient_diagnostics = False

    def _matching_losses(self, inputs: CoEVInputs, corr: dict[str, Tensor], original: Tensor) -> dict[str, Tensor]:
        b = inputs.target_rgb.shape[0]; device = inputs.target_rgb.device
        theta = torch.tensor([[1., .03, .04], [-.02, 1., -.03]], device=device).expand(b, -1, -1)
        grid = F.affine_grid(theta, inputs.target_rgb.shape, align_corners=False)
        warped = F.grid_sample(inputs.target_rgb, grid, align_corners=False)
        warped[..., 144:216, 256:384] = 0
        changed = self.visual_field.encode_measurement_frame(warped)
        pair = self.correspondence_observer(torch.stack((
            _pool_tokens(original.unsqueeze(1), (45, 80))[:, 0],
            _pool_tokens(changed.unsqueeze(1), (45, 80))[:, 0]), 1))
        h, w = 16, 28
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing="ij")
        src = torch.stack((xx, yy), -1).reshape(1,-1,2).expand(b,-1,-1)
        dst = inverse_grid_sample_affine(theta,src)
        col = ((dst[..., 0] + 1) * .5 * (w - 1)).round().long()
        row = ((dst[..., 1] + 1) * .5 * (h - 1)).round().long()
        in_bounds = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        target = (row * w + col).clamp(0, h * w - 1)
        occluded = (dst[..., 0].abs() < .2) & (dst[..., 1].abs() < .2)
        target = torch.where(in_bounds & ~occluded, target, torch.full_like(target, h * w))
        forward_ce = F.nll_loss(pair["forward"][:, 0].clamp_min(1e-8).log().transpose(1, 2), target)
        target_grid=src
        source_from_target=torch.einsum("bij,bnj->bni",theta[:,:,:2],target_grid)+theta[:,:,2].unsqueeze(1)
        reverse_col=((source_from_target[...,0]+1)*.5*(w-1)).round().long();reverse_row=((source_from_target[...,1]+1)*.5*(h-1)).round().long()
        reverse_in=(reverse_col>=0)&(reverse_col<w)&(reverse_row>=0)&(reverse_row<h)
        reverse_target=(reverse_row*w+reverse_col).clamp(0,h*w-1)
        target_occluded=(target_grid[...,0].abs()<.2)&(target_grid[...,1].abs()<.2)
        reverse_target=torch.where(reverse_in&~target_occluded,reverse_target,torch.full_like(reverse_target,h*w))
        reverse_ce=F.nll_loss(pair["reverse"][:,0].clamp_min(1e-8).log().transpose(1,2),reverse_target)
        synthetic_ce=.5*(forward_ce+reverse_ce)
        fxy, fmass, fok = __import__("fate_oia.models.coev_observers", fromlist=["matched_expectation"]).matched_expectation(corr["forward"])
        rxy, _, _ = __import__("fate_oia.models.coev_observers", fromlist=["matched_expectation"]).matched_expectation(corr["reverse"])
        coords = torch.stack((xx, yy), -1).reshape(-1, 2)
        reverse_map = rxy.view(b * 14, 16, 28, 2).permute(0, 3, 1, 2)
        composed = F.grid_sample(reverse_map, fxy.view(b * 14, 16, 28, 2), align_corners=True).permute(0, 2, 3, 1).reshape_as(fxy)
        base = torch.stack((xx, yy), -1).reshape(1, 1, -1, 2)
        cycle = F.smooth_l1_loss(composed, base.expand_as(composed), reduction="none").mean(-1)
        adjacent_valid=inputs.valid[:,:-1]&inputs.valid[:,1:]
        cycle_valid=fok&adjacent_valid.unsqueeze(-1)
        cycle = (cycle * cycle_valid.float()).sum() / cycle_valid.float().sum().clamp_min(1)
        # Adjacent-frame low-resolution photometric consistency at expected coordinates.
        rgb = torch.cat((inputs.history_rgb, F.interpolate(inputs.target_rgb, (256, 448)).unsqueeze(1)), 1)
        small = F.interpolate(rgb.flatten(0, 1), (16, 28), mode="bilinear", align_corners=False).view(b, 15, 3, 16, 28)
        sample_grid = fxy.view(b * 14, 16, 28, 2)
        sampled = F.grid_sample(small[:, 1:].flatten(0, 1), sample_grid, align_corners=True).view(b, 14, 3, 16, 28)
        photo_by_pair=torch.sqrt((sampled-small[:,:-1]).square()+1e-6).mean((2,3,4))
        photo=(photo_by_pair*adjacent_valid.float()).sum()/adjacent_valid.float().sum().clamp_min(1)
        return {"match_synthetic_ce": synthetic_ce, "match_photo": photo, "match_cycle": cycle,
                "match_loss": synthetic_ce + .1 * photo + .1 * cycle}

    def forward(self, inputs: CoEVInputs) -> dict[str, Tensor | int | list[str]]:
        inputs.validate()
        fields = self.visual_field.encode_clip(inputs.target_rgb, inputs.history_rgb, self.chunk_size)
        if self.capture_gradient_diagnostics:
            for key in ("history_task4","history_task8","history_task12"):
                if fields[key].requires_grad: fields[key].retain_grad()
        history_maps = self.predicate_observer(fields["history_low8"], fields["history_grid_hw"])
        target_maps = self.predicate_observer(fields["target_low8"].unsqueeze(1), fields["target_grid_hw"])
        maps = torch.cat((F.adaptive_avg_pool2d(history_maps.flatten(0, 1), (16, 28)).view(*history_maps.shape[:3], 16, 28),
                          F.adaptive_avg_pool2d(target_maps.flatten(0, 1), (16, 28)).view(*target_maps.shape[:3], 16, 28)), 1)
        h_low = _pool_tokens(fields["history_low8"], fields["history_grid_hw"])
        t_low = _pool_tokens(fields["target_low8"].unsqueeze(1), fields["target_grid_hw"])
        corr = self.correspondence_observer(torch.cat((h_low, t_low), 1))
        detached_corr = {key: value.detach() for key, value in corr.items()}
        primitives = self.primitive_builder(maps.detach(), detached_corr, inputs.actual_t, inputs.valid)
        lift = self.path_lift(primitives["primitive_value"], primitives["primitive_valid"], inputs.actual_t)
        decoded = self.video_decoder(fields, inputs.actual_t, inputs.valid, maps.detach())
        readout = self.evidence_readout(decoded["q_video"], lift)
        matching = self._matching_losses(inputs, corr, fields["target_low8"]) if self.training else {"match_loss": readout["logits"].sum() * 0}
        result = {**readout, **primitives, **lift, **matching, "predicate_maps_undetached": maps,
                "matcher_forward": corr["forward"], "matcher_reverse": corr["reverse"],
                "full_history_kv_length": decoded["full_history_kv_length"], "q_video": decoded["q_video"]}
        if self.capture_gradient_diagnostics:
            result["history_gradient_fields"] = tuple(fields[key] for key in ("history_task4","history_task8","history_task12"))
        return result
