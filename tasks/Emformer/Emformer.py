from modules.Emformer.emformer import EmformerDistillModel  
from tasks.Emformer.base_gen_task import AuxDecoderMIDITask
from utils.commons.hparams import hparams
import torch
from utils.commons.ckpt_utils import load_ckpt
from tasks.Emformer.dataset import EmformerDataset
import torch.nn.functional as F
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0
import math
import torch.nn as nn
import random  


def distillation_loss(student_logits: torch.Tensor,
                      teacher_codes: torch.Tensor,
                      mask: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
    """
    Cross-entropy distillation loss for HuBERT codes with masking.

    Args
    ----
    student_logits : (B, T, C) - raw logits from student model
    teacher_codes : (B, T)     - discrete HuBERT indices (targets)
    mask          : (B, T)     - binary mask (1=valid, 0=padding)
    temperature   : float      - temperature scaling for distillation
    """
    B, T, C = student_logits.shape
    
    # Flatten all dimensions
    logits_flat = (student_logits / temperature).view(-1, C)
    target_flat = teacher_codes.reshape(-1).long()
    mask_flat = mask.reshape(-1).bool()
    
    # Only compute loss at masked positions
    if mask_flat.any():
        ce_loss = F.cross_entropy(
            logits_flat[mask_flat], 
            target_flat[mask_flat],
            reduction='mean'
        )
    else:
        ce_loss = torch.tensor(0.0, device=student_logits.device)
    
    return ce_loss


class EmformerTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = EmformerDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        # self.build_disc_model()  # Commented out since discriminator is not implemented
        
        # Distillation parameters (always enabled)
        self.distillation_temperature = hparams.get('distillation_temperature', 1.0)
        self.lambda_distillation = hparams.get('lambda_distillation', 1.0)
        
        # Learning rate warmup parameters
        self.warmup_steps = hparams.get('warmup_steps', 20000)
        self.warmup_init_lr = float(hparams.get('warmup_init_lr', 1e-7))
        self.min_lr = float(hparams.get('min_lr', 1e-6))
        self.lr_decay = hparams.get('lr_decay', 0.999)
        self.decay_interval = hparams.get('decay_interval', 2500)

    def build_tts_model(self):
        # output_dim will be read from hparams in the model
        self.model = EmformerDistillModel(hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    # def build_disc_model(self):
    #     disc_win_num = hparams['disc_win_num']
    #     h = hparams['mel_disc_hidden_size']
    #     self.mel_disc = Discriminator(
    #         time_lengths=[32, 64, 128][:disc_win_num],
    #         freq_length=80, hidden_size=h, kernel=(3, 3)
    #     )
    #     self.disc_params = list(self.mel_disc.parameters())

    def drop_multi(self, tech, drop_p):
        if torch.rand(1) < drop_p:
            tech = torch.ones_like(tech, dtype=tech.dtype) * 2
        elif torch.rand(1) < drop_p:
            random_tech = torch.rand_like(tech, dtype=torch.float32)
            tech[random_tech < drop_p] = 2
        return tech
            
    def run_model(self, sample, infer=False, test=False):
        # txt_tokens = sample["txt_tokens"]
        # mel2ph = sample["mel2ph"]
        # spk_id = sample["spk_ids"]
        content = sample["content"]


        target = sample["mels"]
        
        # Remove the problematic test hack that sets global_step
        # if test:
        #     self.global_step = 200000
        
        # assert False, f'content: {content.shape}, target: {target.shape},spk_embed: {spk_embed.shape}'
        # if not infer:
        #     tech_drop = {
        #         'mix': 0.1,
        #         'falsetto': 0.1,
        #         'breathy': 0.1,
        #         'bubble': 0.1,
        #         'strong': 0.1,
        #         'weak': 0.1,
        #         'glissando': 0.1,
        #         'pharyngeal': 0.1,
        #         'vibrato': 0.1,
        #     }
        #     for tech, drop_p in tech_drop.items():
        #         sample[tech] = self.drop_multi(sample[tech], drop_p)
        
        # mix, falsetto, breathy=sample['mix'], sample['falsetto'], sample['breathy']
        # bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
        # pharyngeal, vibrato, glissando = sample['pharyngeal'], sample['vibrato'], sample['glissando']
        
        # For now, we'll use a simplified approach since EmformerDistillModel has limited interface
        if test or infer:
            # Use the inference method for test/inference
            # The model expects mel input, so we'll use the target mels as input
            # This is a temporary workaround - the model architecture needs to be properly designed
            mel_input = target  # Use target mels as input for now
            lengths = torch.full((mel_input.size(0),), mel_input.size(1), dtype=torch.long, device=mel_input.device)
            
            if hasattr(self.model, 'inference'):
                output = self.model.inference(mel_input)
            else:
                output = self.model(mel_input, lengths)
            
            # Create a compatible output format
            output_dict = {
                'distillation_logits': output  # Always set since distillation is always enabled
            }
            
            # Compute losses for evaluation (no gradients, but needed for metrics)
            losses = {}
            
            # Always compute distillation loss (no mode check needed)
            if 'distillation_logits' in output_dict:
                # Create attention mask from content
                content_attention_mask = (content != -1).float()  # Assuming -1 is padding
                
                # Handle right_context for distillation loss
                # When using streaming models with right_context, we can't predict the last right_context frames
                # because they're used as look-ahead, so we need to truncate accordingly
                # right_context = hparams.get('right_context', 0)
                
                # No right_context, use full sequences
                truncated_content = content
                truncated_mask = content_attention_mask
                truncated_logits = output_dict['distillation_logits']
                
                # Compute distillation loss on truncated sequences
                distill_loss = distillation_loss(
                    truncated_logits, 
                    truncated_content,  # Use truncated content as teacher codes
                    truncated_mask,
                    self.distillation_temperature
                )
                losses['distillation'] = distill_loss * self.lambda_distillation
                
                # Compute distillation accuracy for logging
                accuracy = self.compute_distillation_accuracy(
                    truncated_logits,
                    truncated_content,
                    truncated_mask
                )
                # Store accuracy in output_dict for later logging
                output_dict['distillation_accuracy'] = accuracy
            
            return losses, output_dict
        else:
            # Training mode - use the forward method
            mel_input = target
            lengths = torch.full((mel_input.size(0),), mel_input.size(1), dtype=torch.int64, device=mel_input.device)
            output = self.model(mel_input, lengths)[0] # only take the hubert output
            
            # Create a compatible output format
            output_dict = {
                'distillation_logits': output  # Always set since distillation is always enabled
            }
            
            # Compute losses for training
            losses = {}
            
            # Always compute distillation loss (no mode check needed)
            if 'distillation_logits' in output_dict:
                # Create attention mask from content
                content_attention_mask = (content != -1).float()  # Assuming -1 is padding
                
                # Handle right_context for distillation loss
                # When using streaming models with right_context, we can't predict the last right_context frames
                # because they're used as look-ahead, so we need to truncate accordingly
                right_context = hparams.get('right_context', 0)
                
                if right_context > 0:
                    # Truncate the last right_context frames from targets, mask, and outputs
                    truncated_content = content[:, :-right_context]
                    truncated_mask = content_attention_mask[:, :-right_context]
                    truncated_logits = output_dict['distillation_logits']
                else:
                    # No right_context, use full sequences
                    truncated_content = content
                    truncated_mask = content_attention_mask
                    truncated_logits = output_dict['distillation_logits']
                
                # Compute distillation loss on truncated sequences
                distill_loss = distillation_loss(
                    truncated_logits, 
                    truncated_content,  # Use truncated content as teacher codes
                    truncated_mask,
                    self.distillation_temperature
                )
                losses['distillation'] = distill_loss * self.lambda_distillation
                
                # Compute distillation accuracy for logging
                accuracy = self.compute_distillation_accuracy(
                    truncated_logits,
                    truncated_content,
                    truncated_mask
                )
                # Store accuracy in output_dict for later logging
                output_dict['distillation_accuracy'] = accuracy
            
            return losses, output_dict

            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        # For the basic Emformer model, we only have one optimizer (generator)
        # So we'll ignore optimizer_idx and just run the generator
        if optimizer_idx != 0:
            return None
        loss_output, model_out = self.run_model(sample, infer=False)
        self.model_out_gt = self.model_out = \
            {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
        
        # Add distillation accuracy to loss_output for TensorBoard logging
        if 'distillation_accuracy' in model_out:
            loss_output['distillation_accuracy'] = model_out['distillation_accuracy']
        
        # # Log predicted codes and ground truth codes as text in TensorBoard (occasionally)
        # if (self.global_step % 100 == 0 and  # Log every 100 steps to avoid spam
        #     'distillation_logits' in model_out and 'content' in sample):
        #     content = sample['content']
        #     content_attention_mask = (content != -1).float()
            
        #     # Handle right_context for logging
        #     right_context = hparams.get('right_context', 0)
        #     if right_context > 0:
        #         truncated_content = content[:, :-right_context]
        #         truncated_mask = content_attention_mask[:, :-right_context]
        #         truncated_logits = model_out['distillation_logits']
        #     else:
        #         truncated_content = content
        #         truncated_mask = content_attention_mask
        #         truncated_logits = model_out['distillation_logits']
            
        #     # Log codes as text
        #     self.log_codes_as_text(
        #         truncated_logits, 
        #         truncated_content, 
        #         truncated_mask, 
        #         self.global_step, 
        #         "training"
        #     )
        
        # Simplified training without discriminator
        # The basic Emformer model focuses on mel reconstruction and distillation
        
        total_loss = sum([v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['content'].size()[0]
        return total_loss, loss_output

    def compute_distillation_accuracy(self, student_logits, teacher_codes, mask):
        """
        Compute accuracy for distillation training.
        
        Args:
            student_logits: (B, T, C) - raw logits from student model
            teacher_codes: (B, T) - discrete HuBERT indices (targets)
            mask: (B, T) - binary mask (1=valid, 0=padding)
            
        Returns:
            accuracy: float - accuracy on masked positions
        """
        if student_logits is None or teacher_codes is None:
            return 0.0
        
        # Get predicted codes
        predicted_codes = torch.argmax(student_logits, dim=-1)
        
        # Apply mask
        mask_flat = mask.reshape(-1).bool()
        if not mask_flat.any():
            return 0.0
        
        # Flatten and apply mask
        pred_flat = predicted_codes.reshape(-1)[mask_flat]
        target_flat = teacher_codes.reshape(-1)[mask_flat]
        
        # Compute accuracy
        correct = (pred_flat == target_flat).float().sum()
        total = mask_flat.sum()
        
        return (correct / total).item() if total > 0 else 0.0

    def log_codes_as_text(self, student_logits, teacher_codes, mask, step, prefix="validation"):
        """
        Log predicted codes and ground truth codes as text in TensorBoard.
        
        Args:
            student_logits: (B, T, C) - raw logits from student model
            teacher_codes: (B, T) - discrete HuBERT indices (targets)
            mask: (B, T) - binary mask (1=valid, 0=padding)
            step: int - current step for TensorBoard logging
            prefix: str - prefix for TensorBoard tags
        """
        if student_logits is None or teacher_codes is None:
            return
        
        # Get predicted codes
        predicted_codes = torch.argmax(student_logits, dim=-1)
        
        # Apply mask to get valid positions
        mask_bool = mask.bool()
        
        # For each batch item, create text representation
        for batch_idx in range(min(2, predicted_codes.size(0))):  # Log first 2 batch items
            # Get valid positions for this batch item
            valid_positions = mask_bool[batch_idx]
            
            if not valid_positions.any():
                continue
            
            # Get codes for valid positions only
            pred_codes_valid = predicted_codes[batch_idx][valid_positions]
            target_codes_valid = teacher_codes[batch_idx][valid_positions]
            
            # Convert to text format: "71 23 35..."
            pred_text = " ".join([str(code.item()) for code in pred_codes_valid])
            target_text = " ".join([str(code.item()) for code in target_codes_valid])
            
            # Log to TensorBoard as text
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.add_text(
                    f"{prefix}/predicted_codes_batch_{batch_idx}", 
                    pred_text, 
                    global_step=step
                )
                self.logger.add_text(
                    f"{prefix}/ground_truth_codes_batch_{batch_idx}", 
                    target_text, 
                    global_step=step
                )
    def validation_start(self):
        pass
    
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['nsamples'] = sample['nsamples']
        
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        
        # Add distillation accuracy to losses for TensorBoard logging
        if 'distillation_accuracy' in model_out:
            outputs['losses']['distillation_accuracy'] = model_out['distillation_accuracy']
        
        # Log predicted codes and ground truth codes as text in TensorBoard
        if 'distillation_logits' in model_out and 'content' in sample:
            content = sample['content']
            content_attention_mask = (content != -1).float()
            
            truncated_content = content
            truncated_mask = content_attention_mask
            truncated_logits = model_out['distillation_logits']
            
            # Log codes as text
            self.log_codes_as_text(
                truncated_logits, 
                truncated_content, 
                truncated_mask, 
                self.global_step, 
                "validation"
            )
        
        outputs['total_loss'] = sum([v for k, v in outputs['losses'].items() 
                                    if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and not v.requires_grad)])
        outputs = tensors_to_scalars(outputs)
        return outputs
    
    def test_step(self, sample, batch_idx):
        """
        Simplified test step that leverages EmformerDistillModel's internal streaming capabilities.
        The model handles chunking and streaming internally, so we just need to call inference.
        """
        # Prepare the sample for inference
        sample['ref_mels'] = sample['mels']  # Keep consistent with original implementation
        
        # Run the model with inference mode
        losses, outputs = self.run_model(sample, infer=True, test=True)
        
        # Log predicted codes and ground truth codes as text in TensorBoard
        if 'distillation_logits' in outputs and 'content' in sample:
            content = sample['content']
            content_attention_mask = (content != -1).float()
            
            # Handle right_context for logging
            right_context = hparams.get('right_context', 0)
            if right_context > 0:
                truncated_content = content[:, :-right_context]
                truncated_mask = content_attention_mask[:, :-right_context]
                truncated_logits = outputs['distillation_logits']
            else:
                truncated_content = content
                truncated_mask = content_attention_mask
                truncated_logits = outputs['distillation_logits']
            
            # Log codes as text
            self.log_codes_as_text(
                truncated_logits, 
                truncated_content, 
                truncated_mask, 
                self.global_step, 
                "test"
            )
        
        # Extract the hubert output
        # Prepare for saving
        item_name = sample['item_name'][0]
        base_fn = f'{item_name.replace(" ", "_")}[P]'
        
        
        # Save ground truth if requested
        if hparams.get('save_gt', False):
            mel_gt = sample['mels'][0].cpu().numpy()
            wav_gt = self.vocoder.spec2wav(mel_gt)  # Removed F0 since we don't have it
            
            self.saving_result_pool.add_job(
                self.save_result,
                args=[wav_gt, mel_gt,
                    f'{item_name.replace(" ", "_")}[G]', self.gen_dir, None, None,
                    None, None, None]  # Removed F0-related arguments
            )
        
        return {}


    def build_optimizer(self, model):
        
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        # optimizer_disc = torch.optim.AdamW(
        #     self.disc_params,
        #     lr=hparams['disc_lr'],
        #     betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
        #     **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen]

    def build_scheduler(self, optimizer):
        # Always use custom warmup + decay scheduler for distillation training
        return [self.build_lr_scheduler(optimizer[0])]
    
    def build_lr_scheduler(self, optimizer):
        """
        Build learning rate scheduler with warmup and decay.
        This replaces the default scheduler for distillation training.
        """
        # Always use custom warmup + decay scheduler for distillation
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warm-up
                warmup_frac = step / float(self.warmup_steps)
                return (self.warmup_init_lr / hparams['lr']) + (1 - self.warmup_init_lr / hparams['lr']) * warmup_frac
            else:
                # Step-based decay
                decay_steps = (step - self.warmup_steps) // self.decay_interval
                new_lr = hparams['lr'] * (self.lr_decay ** decay_steps)
                return max(new_lr, self.min_lr) / hparams['lr']
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    def on_before_optimization(self, opt_idx):
        nn.utils.clip_grad_norm_(self.gen_params, hparams['clip_grad_norm'])
        # else:
        #     nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            # Always step the scheduler every step for distillation training
            self.scheduler[0].step(self.global_step)


def self_clone(x):
    if x == None:
        return None
    y = x.clone()
    result = torch.cat((x, y), dim=0)
    return result
