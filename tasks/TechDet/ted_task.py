import os
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchmetrics.functional.classification import binary_auroc, binary_recall, binary_f1_score, binary_precision, binary_accuracy
from torchmetrics import AUROC, Recall, F1Score, Precision, Accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import filecmp
import pandas as pd

import utils
from utils import seed_everything
from utils.commons.hparams import hparams
from tasks.tts.tts_utils import parse_dataset_configs
from utils.audio.pitch_utils import denorm_f0
from utils.commons.tensor_utils import tensors_to_scalars
from utils.commons.ckpt_utils import load_ckpt
from utils.nn.model_utils import print_arch
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.commons.dataset_utils import data_loader, BaseConcatDataset
from utils.commons.losses import sigmoid_focal_loss

from tasks.tts.speech_base import SpeechBaseTask
from tasks.TechDet.ted_dataset import TEDataset
from modules.TechDet.ted import TechExtractor


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure(figsize=(12, 8))
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='ref')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, labels=None, get_label_fn=None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))
        # define custom callback
        self.num_samples = len(self.indices)
        self.get_label_fn = get_label_fn
        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()
        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        indices = dataset.ordered_indices()
        if self.get_label_fn != None:
            return [self.get_label_fn(id) for id in indices]
        elif dataset.get_label != None:
            return [dataset.get_label(id) for id in indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class TETask(SpeechBaseTask):
    def __init__(self, *args, **kwargs):
        super(SpeechBaseTask, self).__init__(*args, **kwargs)
        self.dataset_cls = TEDataset
        self.vocoder = None
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.max_tokens, self.max_sentences, \
            self.max_valid_tokens, self.max_valid_sentences = parse_dataset_configs()
        seed_everything(hparams['seed'])

        # UserWarning: No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score
        warnings.filterwarnings("ignore", category=UserWarning)

    @data_loader
    def train_dataloader(self):
        if hparams['train_sets'] != '':
            train_sets = hparams['train_sets'].split("|")
            # check if all train_sets have the same spk map and dictionary
            binary_data_dir = hparams['binary_data_dir']
            file_to_cmp = ['phone_set.json']
            if os.path.exists(f'{binary_data_dir}/word_set.json'):
                file_to_cmp.append('word_set.json')
            if hparams['use_spk_id']:
                file_to_cmp.append('spk_map.json')
            for f in file_to_cmp:
                for ds_name in train_sets:
                    base_file = os.path.join(binary_data_dir, f)
                    ds_file = os.path.join(ds_name, f)
                    assert filecmp.cmp(base_file, ds_file), \
                        f'{f} in {ds_name} is not same with that in {binary_data_dir}.'
            train_dataset = BaseConcatDataset([
                self.dataset_cls(prefix='train', shuffle=True, data_dir=ds_name) for ds_name in train_sets])
        else:
            train_dataset = self.dataset_cls(prefix=hparams['train_set_name'], shuffle=True)

        if hparams.get('apply_weighted_sampler', True):
            print('| Applying weighted sampler.')
            return self.build_dataloader(train_dataset, False, self.max_tokens, self.max_sentences, sample_by_weight=True,
                                         endless=hparams['endless_ds'], pin_memory=hparams.get('pin_memory', False))
        else:
            return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'], pin_memory=hparams.get('pin_memory', False))

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None, required_batch_size_multiple=-1,
                         endless=False, batch_by_size=True, pin_memory=False, sample_by_weight=False, ):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        sampler = None
        if sample_by_weight:
            sampler = ImbalancedDatasetSampler(dataset)

        if batch_by_size:
            batch_sampler = utils.commons.dataset_utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        if sampler != None:
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=max_sentences,
                                               sampler=sampler,
                                               collate_fn=dataset.collater,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)
        else:
            return torch.utils.data.DataLoader(dataset,
                                               collate_fn=dataset.collater,
                                               batch_sampler=batches,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            load_ckpt(self.model, hparams['load_ckpt'])
        print_arch(self.model)
        return self.model

    def build_tts_model(self):
        model_name = hparams.get('model', None)
        self.model = TechExtractor(hparams)

    def _training_step(self, sample, batch_idx, _):
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output

    def run_model(self, sample, infer=False):
        mel = sample['mels']
        pitch_coarse = sample['pitch_coarse']
        uv = sample['uv'].long()
        mel_nonpadding = sample['mel_nonpadding']
        ph_bd = sample['ph_bd']
        breathiness = sample.get('breathiness', None)
        energy = sample.get('energy', None)
        zcr = sample.get('zcr', None)
        variance = {'breathiness': breathiness, 'energy': energy, 'zcr': zcr}
        output = self.model(mel=mel, ph_bd=ph_bd, pitch=pitch_coarse, uv=uv, variance=variance,
                            non_padding=mel_nonpadding, train=not infer)
        losses = {}
        if not infer:
            techs = sample['techs']
            tech_ids = sample['tech_ids']
            if hparams.get('apply_tech_group_loss', False) and tech_ids is not None:
                self.add_tech_group_loss(output['tech_logits'], output['mf_tech_logits'], output['sw_tech_logits'], techs, tech_ids, losses, sample)
            else:
                self.add_tech_loss(output['tech_logits'], output['mf_tech_logits'], output['sw_tech_logits'], techs, losses)
        return losses, output
        
    def add_tech_group_loss(self, tech_logits, mf_tech_logits, sw_tech_logits, techs, tech_ids, losses, sample=None):
        bsz, T, binary_num_techs = tech_logits.shape
        num_techs= techs.shape[2]
        assert binary_num_techs + 2==num_techs ,f'binary_num_techs:{binary_num_techs}, num_techs:{num_techs}'
        tech_losses = torch.zeros([T, num_techs], device=tech_logits.device)
        for b_idx in range(bsz):
            # NOTE: tech_ids is a index list
            if tech_ids[b_idx] < binary_num_techs:
                tech_losses_i = F.binary_cross_entropy_with_logits(tech_logits[b_idx, :, tech_ids[b_idx]], techs[b_idx, :, tech_ids[b_idx]].float(), reduction='none')  # [T, len(tech_ids)]
            elif tech_ids[b_idx]==5:
                tech_losses_i = F.cross_entropy(mf_tech_logits[b_idx], techs[b_idx, :, tech_ids[b_idx]],
                                            label_smoothing=hparams.get('mf_tech_label_smoothing', 0.0), reduction='none')
            elif tech_ids[b_idx]==6:
                tech_losses_i = F.cross_entropy(sw_tech_logits[b_idx], techs[b_idx, :, tech_ids[b_idx]],
                                            label_smoothing=hparams.get('sw_tech_label_smoothing', 0.0), reduction='none')
                # accuracy = Accuracy(task="multiclass", num_classes=3).to(tech_losses_i.device)
                # temp_a = accuracy(sw_tech_logits[b_idx].unsqueeze(0).transpose(1, 2), techs[b_idx, :, tech_ids[b_idx]].unsqueeze(0))
                # print(temp_a)
            
            tech_losses[:, tech_ids[b_idx]] += tech_losses_i    # [T, C]
        tech_losses /= bsz
        tech_losses = tech_losses.mean(0)
        lambda_tech = hparams.get('lambda_tech', 1.0)
        lambdas_tech = hparams.get('lambdas_tech', '')
        if lambdas_tech != '' and '-' in lambdas_tech:
            lambda_tech = [float(i) for i in lambdas_tech.split('-')]
            assert len(lambda_tech) == num_techs, f"{len(lambda_tech)} {num_techs}"
        else:
            lambda_tech = [lambda_tech for _ in range(num_techs)]
        losses['bubble'] = tech_losses[0] * lambda_tech[0]
        losses['breathe'] = tech_losses[1] * lambda_tech[1]
        losses['pharyngeal'] = tech_losses[2] * lambda_tech[2]
        losses['vibrato'] = tech_losses[3] * lambda_tech[3]
        losses['glissando'] = tech_losses[4] * lambda_tech[4]
        losses['mf_tech'] = tech_losses[5] * lambda_tech[5]
        losses['sw_tech'] = tech_losses[6] * lambda_tech[6]

    def add_tech_loss(self, tech_logits, mf_tech_logits, sw_tech_logits, techs, losses):
        bsz, T, binary_num_techs = tech_logits.shape
        assert binary_num_techs==5 ,f'binary_num_techs:{binary_num_techs}'
        tech_losses = F.binary_cross_entropy_with_logits(tech_logits, techs[:, :, :binary_num_techs].float(), reduction='none')  # [B, T, C]
        tech_losses = tech_losses.reshape(-1, binary_num_techs).mean(0)    # [C]
        lambda_tech = hparams.get('lambda_tech', 1.0)
        lambdas_tech = hparams.get('lambdas_tech', '')
        if lambdas_tech != '' and '-' in lambdas_tech:
            lambda_tech = [float(i) for i in lambdas_tech.split('-')]
            assert len(lambda_tech) == binary_num_techs+2, f"{len(lambda_tech)} {binary_num_techs+2}"
        else:
            lambda_tech = [lambda_tech for _ in range(binary_num_techs+2)]
        losses['bubble'] = tech_losses[0] * lambda_tech[0]
        losses['breathe'] = tech_losses[1] * lambda_tech[1]
        losses['pharyngeal'] = tech_losses[2] * lambda_tech[2]
        losses['vibrato'] = tech_losses[3] * lambda_tech[3]
        losses['glissando'] = tech_losses[4] * lambda_tech[4]
        losses['mf_tech'] = F.cross_entropy(mf_tech_logits.transpose(1, 2), techs[:, :, 5], 
                        label_smoothing=hparams.get('mf_tech_label_smoothing', 0.0)) * lambda_tech[5]
        losses['sw_tech'] = F.cross_entropy(sw_tech_logits.transpose(1, 2), techs[:, :, 6],
                        label_smoothing=hparams.get('sw_tech_label_smoothing', 0.0)) * lambda_tech[6]
        if hparams.get('tech_focal_loss', None) not in ['none', None, 0]:
            gamma = float(hparams.get('tech_focal_loss', None))
            focal_loss = sigmoid_focal_loss(tech_logits, techs[:, :, :binary_num_techs].float(), alpha=-1, gamma=gamma, reduction='mean')
            losses['tech_fc'] = focal_loss * hparams.get('lambda_tech_focal', 1.0)

    def validation_start(self):
        self.vocoder = None

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        with torch.no_grad():
            outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        if batch_idx < hparams['num_valid_stats']:
            mel = sample['mels']
            pitch_coarse = sample['pitch_coarse']
            uv = sample['uv'].long()
            mel_nonpadding = sample['mel_nonpadding']
            breathiness = sample.get('breathiness', None)
            energy = sample.get('energy', None)
            zcr = sample.get('zcr', None)
            variance = {'breathiness': breathiness, 'energy': energy, 'zcr': zcr}
            ph_bd = sample['ph_bd']
            techs = sample['techs']
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])

            with torch.no_grad():
                # no note_bd
                output = self.model(mel=mel, ph_bd=ph_bd, pitch=pitch_coarse, uv=uv, variance=variance,
                                    non_padding=mel_nonpadding, train=True)
                tech_logits = output['tech_logits']
                mf_tech_logits = output['mf_tech_logits']
                sw_tech_logits = output['sw_tech_logits']
                num_techs = techs.shape[2]
                binary_num_techs = tech_logits.shape[2]
                assert binary_num_techs+2==num_techs, f'binary_num_techs:{binary_num_techs}, num_techs:{num_techs}'
                tech_probs = torch.sigmoid(tech_logits)
                tech_pred = output['tech_pred']
                threshold = hparams.get('tech_threshold', 0.8)
                
                auroc = AUROC(task="multiclass", num_classes=3).to(sw_tech_logits.device)
                recall = Recall(task="multiclass", num_classes=3).to(sw_tech_logits.device)
                f1score = F1Score(task="multiclass", num_classes=3).to(sw_tech_logits.device)
                precision = Precision(task="multiclass", num_classes=3).to(sw_tech_logits.device)
                accuracy = Accuracy(task="multiclass", num_classes=3).to(sw_tech_logits.device)
                tech_names = ['bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mf', 'sw']
                valid_tech_num = 0
                outputs['losses']['tech_auroc'] = 0
                outputs['losses']['tech_p'] = 0
                outputs['losses']['tech_r'] = 0
                outputs['losses']['tech_f'] = 0
                outputs['losses']['tech_a'] = 0
                for tech_idx, tech_name in enumerate(tech_names):
                    if tech_idx < binary_num_techs: 
                        if torch.sum(techs[:, :, tech_idx]) > 0:
                            outputs['losses'][f'{tech_name}_auroc'] = binary_auroc(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                            outputs['losses'][f'{tech_name}_p'] = binary_precision(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                            outputs['losses'][f'{tech_name}_r'] = binary_recall(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                            outputs['losses'][f'{tech_name}_f'] = binary_f1_score(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                            outputs['losses'][f'{tech_name}_a'] = binary_accuracy(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                            valid_tech_num += 1
                    else:
                        if tech_idx==5:
                            multi_tech_logits = mf_tech_logits.transpose(1, 2)
                        elif tech_idx==6:
                            multi_tech_logits = sw_tech_logits.transpose(1, 2)
                        if torch.sum(techs[:, :, tech_idx]) > 0:
                            outputs['losses'][f'{tech_name}_auroc'] = auroc(multi_tech_logits, techs[:, :, tech_idx])
                            outputs['losses'][f'{tech_name}_p'] = precision(multi_tech_logits, techs[:, :, tech_idx])
                            outputs['losses'][f'{tech_name}_r'] = recall(multi_tech_logits, techs[:, :, tech_idx])
                            outputs['losses'][f'{tech_name}_f'] = f1score(multi_tech_logits, techs[:, :, tech_idx])
                            outputs['losses'][f'{tech_name}_a'] = accuracy(multi_tech_logits, techs[:, :, tech_idx])
                            valid_tech_num += 1
                    outputs['losses']['tech_auroc'] += outputs['losses'][f'{tech_name}_auroc'] if f'{tech_name}_auroc' in outputs['losses'] else 0
                    outputs['losses']['tech_p'] += outputs['losses'][f'{tech_name}_p'] if f'{tech_name}_p' in outputs['losses'] else 0
                    outputs['losses']['tech_r'] += outputs['losses'][f'{tech_name}_r'] if f'{tech_name}_r' in outputs['losses'] else 0
                    outputs['losses']['tech_f'] += outputs['losses'][f'{tech_name}_f'] if f'{tech_name}_f' in outputs['losses'] else 0
                    outputs['losses']['tech_a'] += outputs['losses'][f'{tech_name}_a'] if f'{tech_name}_a' in outputs['losses'] else 0
                    
                    if batch_idx < hparams['num_valid_plots']: 
                        single_techs = techs[0, :, tech_idx]                       
                        if tech_idx < binary_num_techs:
                            single_tech_pred = tech_pred[0, :, tech_idx]
                            single_tech_probs = tech_probs[0, :, tech_idx]
                        elif tech_idx==5:
                            single_tech_pred = output['mf_tech_pred'][0, :]
                            single_tech_probs = torch.softmax(mf_tech_logits, dim=-1)
                            single_tech_probs = torch.max(single_tech_probs, dim=-1)[0][0]
                        elif tech_idx==6:
                            single_tech_pred = output['sw_tech_pred'][0, :]
                            single_tech_probs = torch.softmax(sw_tech_logits, dim=-1)
                            single_tech_probs = torch.max(single_tech_probs, dim=-1)[0][0]
                        self.logger.add_figure(
                            f'tech_{tech_name}_{batch_idx}',
                            f0_tech_to_figure(gt_f0[0].data.cpu().numpy(), ph_bd[0].data.cpu().numpy(),
                                              single_tech_pred.data.cpu().numpy(),
                                              single_techs.data.cpu().numpy(),
                                              single_tech_probs.data.cpu().numpy(),
                                              tech_name, fig_name=sample['item_name']),
                            self.global_step)
                
                if valid_tech_num != 0:
                    outputs['losses']['tech_a'] /= valid_tech_num
                    outputs['losses']['tech_auroc'] /= valid_tech_num
                    outputs['losses']['tech_p'] /= valid_tech_num
                    outputs['losses']['tech_r'] /= valid_tech_num
                    outputs['losses']['tech_f'] /= valid_tech_num
                else:
                    del outputs['losses']['tech_auroc']
                    del outputs['losses']['tech_p']
                    del outputs['losses']['tech_r']
                    del outputs['losses']['tech_f']

            self.save_valid_result(sample, batch_idx, model_out)
        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        # torch.cuda.empty_cache()
        return super(TETask, self).validation_end(outputs)

    def save_valid_result(self, sample, batch_idx, model_out):
        pass

    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)

    # def test_step(self, sample, batch_idx):
    #     _, outputs = self.run_model(sample, infer=True)
    #     ph_bd = sample['ph_bd'][0].cpu()
    #     tech_gt = sample['techs'][0].cpu()
    #     f0 = denorm_f0(sample['f0'], sample['uv'])[0].cpu()
    #     tech_logits = outputs['tech_logits'][0].cpu()
    #     tech_probs = torch.sigmoid(outputs['tech_logits'])[0].cpu()
    #     tech_pred = outputs['tech_pred'][0].cpu()
    #     threshold = hparams.get('tech_threshold', 0.8)
    #     mf_tech_logits = outputs['mf_tech_logits'][0].cpu()
    #     sw_tech_logits = outputs['sw_tech_logits'][0].cpu()
    #     mf_tech_pred = outputs['mf_tech_pred'][0].cpu()
    #     sw_tech_pred = outputs['sw_tech_pred'][0].cpu()
    #     item_name = sample['item_name'][0]
    #     gen_dir = self.gen_dir
    #     self.saving_result_pool.add_job(self.save_result, args=[
    #         item_name, gen_dir, f0, ph_bd, tech_gt, tech_probs, tech_pred, tech_logits, mf_tech_logits, sw_tech_logits, mf_tech_pred, sw_tech_pred, threshold])
    #     return {}

    # @staticmethod
    # def save_result(item_name, gen_dir, gt_f0=None, ph_bd=None, tech_gt=None, tech_probs=None, tech_pred=None,
    #                 tech_logits=None, mf_tech_logits=None, sw_tech_logits=None, mf_tech_pred=None, sw_tech_pred=None, threshold=0.8):
    #     auroc = AUROC(task="multiclass", num_classes=3).to(sw_tech_logits.device)
    #     recall = Recall(task="multiclass", num_classes=3).to(sw_tech_logits.device)
    #     f1score = F1Score(task="multiclass", num_classes=3).to(sw_tech_logits.device)
    #     precision = Precision(task="multiclass", num_classes=3).to(sw_tech_logits.device)
    #     accuracy = Accuracy(task="multiclass", num_classes=3).to(sw_tech_logits.device)
    #     tech_names = ['bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mf', 'sw']
    #     valid_tech_num = 0
    #     binary_num_techs = 5
    #     overall_tech_auroc = 0
    #     overall_tech_p = 0
    #     overall_tech_r = 0
    #     overall_tech_f = 0
    #     overall_tech_a = 0
    #     res = []
        
    #     for tech_idx, tech_name in enumerate(tech_names):
    #         # print(tech_idx, tech_name)
    #         tech_auroc = np.nan
    #         tech_p = np.nan
    #         tech_r = np.nan
    #         tech_f = np.nan
    #         tech_a = np.nan
    #         if tech_idx < binary_num_techs and torch.sum(tech_gt[:, tech_idx]) > 0:
    #             tech_auroc = binary_auroc(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
    #             tech_p = binary_precision(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
    #             tech_r = binary_recall(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
    #             tech_f = binary_f1_score(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
    #             tech_a = binary_accuracy(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold) 
    #             valid_tech_num += 1                   
    #         else:
    #             if tech_idx==5:
    #                 multi_tech_logits = mf_tech_logits 
    #             elif tech_idx==6:
    #                 multi_tech_logits = sw_tech_logits
    #             if torch.sum(tech_gt[:, tech_idx]) > 0:
    #                 tech_auroc = auroc(multi_tech_logits, tech_gt[:, tech_idx])
    #                 tech_p = precision(multi_tech_logits, tech_gt[:, tech_idx])
    #                 tech_r = recall(multi_tech_logits, tech_gt[:, tech_idx])
    #                 tech_f = f1score(multi_tech_logits, tech_gt[:, tech_idx])
    #                 tech_a = accuracy(multi_tech_logits, tech_gt[:, tech_idx])
    #                 valid_tech_num += 1
            
    #         res = res + [tech_p, tech_r, tech_f, tech_a, tech_auroc]
    #         overall_tech_auroc += tech_auroc if not np.isnan(tech_auroc) else 0
    #         overall_tech_p += tech_p if not np.isnan(tech_p) else 0
    #         overall_tech_r += tech_r if not np.isnan(tech_r) else 0
    #         overall_tech_f += tech_f if not np.isnan(tech_f) else 0
    #         overall_tech_a += tech_a if not np.isnan(tech_a) else 0
    #         single_techs = tech_gt[:, tech_idx]                       
    #         if tech_idx < binary_num_techs:
    #             single_tech_pred = tech_pred[:, tech_idx]
    #             single_tech_probs = tech_probs[:, tech_idx].numpy()
    #         elif tech_idx==5:
    #             single_tech_pred = mf_tech_pred
    #             tri_tech_probs = torch.softmax(mf_tech_logits, dim=-1)
    #             single_tech_probs = np.max(tri_tech_probs.cpu().numpy(), axis=1)
    #         elif tech_idx==6:
    #             single_tech_pred = sw_tech_pred
    #             single_tech_probs = torch.softmax(sw_tech_logits, dim=-1)
    #             single_tech_probs = np.max(tri_tech_probs.cpu().numpy(), axis=1)
    #         fig = f0_tech_to_figure(gt_f0.numpy(), ph_bd.numpy(),
    #                             single_tech_pred.numpy(),
    #                             single_techs.numpy(),
    #                             single_tech_probs,
    #                             tech_name, fig_name=item_name,
    #                             save_path=f'{gen_dir}/plot/{item_name}[{tech_name}].png')
    #         # print('yes')
    #         plt.close(fig)
    #     if valid_tech_num != 0:
    #         overall_tech_a /= valid_tech_num
    #         overall_tech_auroc /= valid_tech_num
    #         overall_tech_p /= valid_tech_num
    #         overall_tech_r /= valid_tech_num
    #         overall_tech_f /= valid_tech_num
    #         # print(overall_tech_a)
    #     else:
    #         overall_tech_a = np.nan
    #         overall_tech_auroc = np.nan
    #         overall_tech_p = np.nan
    #         overall_tech_r = np.nan
    #         overall_tech_f = np.nan
    #     res = [overall_tech_p, overall_tech_r, overall_tech_f, overall_tech_a, overall_tech_auroc] + res
    #     gender = 'male' if '男声' in item_name or 'Tenor' in item_name else 'female'
    #     return gender, res

    def test_step(self, sample, batch_idx):
        _, outputs = self.run_model(sample, infer=True)
        ph_bd = sample['ph_bd'][0].cpu().numpy()
        tech_gt = sample['techs'][0].cpu().numpy()
        f0 = denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy()
        tech_logits = outputs['tech_logits'][0].cpu().numpy()
        tech_probs = torch.sigmoid(outputs['tech_logits'])[0].cpu().numpy()
        tech_pred = outputs['tech_pred'][0].cpu().numpy()
        threshold = hparams.get('tech_threshold', 0.8)
        mf_tech_logits = outputs['mf_tech_logits'][0].cpu().numpy()
        sw_tech_logits = outputs['sw_tech_logits'][0].cpu().numpy()
        mf_tech_pred = outputs['mf_tech_pred'][0].cpu().numpy()
        sw_tech_pred = outputs['sw_tech_pred'][0].cpu().numpy()
        item_name = sample['item_name'][0]
        gen_dir = self.gen_dir
        self.saving_result_pool.add_job(self.save_result, args=[
            item_name, gen_dir, f0, ph_bd, tech_gt, tech_probs, tech_pred, tech_logits, 
            mf_tech_logits, sw_tech_logits, mf_tech_pred, sw_tech_pred, threshold])
        return {}

    @staticmethod
    def save_result(item_name, gen_dir, gt_f0=None, ph_bd=None, tech_gt=None, tech_probs=None, tech_pred=None,
                    tech_logits=None, mf_tech_logits=None, sw_tech_logits=None, mf_tech_pred=None, sw_tech_pred=None, threshold=0.8):
        # Convert numpy arrays back to tensors for metric computation
        tech_gt = torch.from_numpy(tech_gt)
        tech_logits = torch.from_numpy(tech_logits)
        mf_tech_logits = torch.from_numpy(mf_tech_logits)
        sw_tech_logits = torch.from_numpy(sw_tech_logits)
        mf_tech_pred = torch.from_numpy(mf_tech_pred)
        sw_tech_pred = torch.from_numpy(sw_tech_pred)
        gt_f0 = torch.from_numpy(gt_f0)
        ph_bd = torch.from_numpy(ph_bd)
        tech_probs = torch.from_numpy(tech_probs)
        tech_pred = torch.from_numpy(tech_pred)

        auroc = AUROC(task="multiclass", num_classes=3)
        recall = Recall(task="multiclass", num_classes=3)
        f1score = F1Score(task="multiclass", num_classes=3)
        precision = Precision(task="multiclass", num_classes=3)
        accuracy = Accuracy(task="multiclass", num_classes=3)
        tech_names = ['bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mf', 'sw']
        valid_tech_num = 0
        binary_num_techs = 5
        overall_tech_auroc = 0.0
        overall_tech_p = 0.0
        overall_tech_r = 0.0
        overall_tech_f = 0.0
        overall_tech_a = 0.0
        res = []
        
        for tech_idx, tech_name in enumerate(tech_names):
            tech_auroc = np.nan
            tech_p = np.nan
            tech_r = np.nan
            tech_f = np.nan
            tech_a = np.nan
            if tech_idx < binary_num_techs and torch.sum(tech_gt[:, tech_idx]) > 0:
                # Binary metrics
                tech_auroc = binary_auroc(torch.tensor(tech_logits[:, tech_idx]), tech_gt[:, tech_idx], threshold).item()
                tech_p = binary_precision(torch.tensor(tech_logits[:, tech_idx]), tech_gt[:, tech_idx], threshold).item()
                tech_r = binary_recall(torch.tensor(tech_logits[:, tech_idx]), tech_gt[:, tech_idx], threshold).item()
                tech_f = binary_f1_score(torch.tensor(tech_logits[:, tech_idx]), tech_gt[:, tech_idx], threshold).item()
                tech_a = binary_accuracy(torch.tensor(tech_logits[:, tech_idx]), tech_gt[:, tech_idx], threshold).item()
                valid_tech_num += 1
            else:
                if tech_idx == 5:
                    multi_tech_logits = mf_tech_logits
                elif tech_idx == 6:
                    multi_tech_logits = sw_tech_logits
                if torch.sum(tech_gt[:, tech_idx]) > 0:
                    tech_auroc = auroc(multi_tech_logits, tech_gt[:, tech_idx]).item()
                    tech_p = precision(multi_tech_logits, tech_gt[:, tech_idx]).item()
                    tech_r = recall(multi_tech_logits, tech_gt[:, tech_idx]).item()
                    tech_f = f1score(multi_tech_logits, tech_gt[:, tech_idx]).item()
                    tech_a = accuracy(multi_tech_logits, tech_gt[:, tech_idx]).item()
                    valid_tech_num += 1
            
            res += [tech_p, tech_r, tech_f, tech_a, tech_auroc]
            overall_tech_auroc += tech_auroc if not np.isnan(tech_auroc) else 0.0
            overall_tech_p += tech_p if not np.isnan(tech_p) else 0.0
            overall_tech_r += tech_r if not np.isnan(tech_r) else 0.0
            overall_tech_f += tech_f if not np.isnan(tech_f) else 0.0
            overall_tech_a += tech_a if not np.isnan(tech_a) else 0.0
            
            # Plotting logic (unchanged but ensure numpy arrays are used)
            if tech_idx < binary_num_techs:
                single_tech_probs = tech_probs[:, tech_idx].numpy()
                single_tech_pred = tech_pred[:, tech_idx].numpy()
            elif tech_idx == 5:
                tri_tech_probs = torch.softmax(mf_tech_logits, dim=-1).numpy()
                single_tech_probs = np.max(tri_tech_probs, axis=1)
                single_tech_pred = mf_tech_pred.numpy()
            elif tech_idx == 6:
                tri_tech_probs = torch.softmax(sw_tech_logits, dim=-1).numpy()
                single_tech_probs = np.max(tri_tech_probs, axis=1)
                single_tech_pred = sw_tech_pred.numpy()
            single_techs = tech_gt[:, tech_idx].numpy()
            
            fig = f0_tech_to_figure(gt_f0.numpy(), ph_bd.numpy(),
                                    single_tech_pred,
                                    single_techs,
                                    single_tech_probs,
                                    tech_name, fig_name=item_name,
                                    save_path=f'{gen_dir}/plot/{item_name}[{tech_name}].png')
            plt.close(fig)
        
        if valid_tech_num != 0:
            overall_tech_a /= valid_tech_num
            overall_tech_auroc /= valid_tech_num
            overall_tech_p /= valid_tech_num
            overall_tech_r /= valid_tech_num
            overall_tech_f /= valid_tech_num
        else:
            overall_tech_a = np.nan
            overall_tech_auroc = np.nan
            overall_tech_p = np.nan
            overall_tech_r = np.nan
            overall_tech_f = np.nan
        
        res = [overall_tech_p, overall_tech_r, overall_tech_f, overall_tech_a, overall_tech_auroc] + res
        gender = 'male' if '男声' in item_name or 'Tenor' in item_name else 'female'
        return gender, res

    def test_end(self, outputs):
        res_dict = {'male': [], 'female': [], 'overall': []}
        for r_id, r in tqdm(self.saving_result_pool.get_results(), total=len(self.saving_result_pool)):
            res_dict[r[0]].append(r[1])
            res_dict['overall'].append(r[1])

        for gender, res in res_dict.items():
            print("=" * 20 + f" {gender} " + "=" * 20)
            res = np.array(res)
            results = []
            print(res.shape)
            for i in range(res.shape[1]):
                _res = res[:, i]
                results.append(np.mean(_res[~np.isnan(_res)]))

            tech_names = ['overall', 'bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mf', 'sw']
            for tech_idx, tech_name in enumerate(tech_names):
                print(f"{tech_name}: precision: {results[5 * tech_idx]:.3f}, recall: {results[5 * tech_idx + 1]:.3f}, f1: {results[5 * tech_idx + 2]:.3f}, accuracy: {results[5 * tech_idx + 3]:.3f}, auroc: {results[5 * tech_idx + 4]:.3f}")


class TEODTask(TETask):
    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_od_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)

    def test_step(self, sample, batch_idx):
        _, outputs = self.run_model(sample, infer=True)
        ph_bd = sample['ph_bd'][0].cpu()
        ph = sample['ph'][0]
        # tech_gt = sample['techs'][0].cpu()
        f0 = denorm_f0(sample['f0'], sample['uv'])[0].cpu()
        tech_logits = outputs['tech_logits'][0].cpu()
        tech_probs = torch.sigmoid(outputs['tech_logits'])[0].cpu()
        tech_pred = outputs['tech_pred'][0].cpu()
        threshold = hparams.get('tech_threshold', 0.8)

        item_name = sample['item_name'][0]
        gen_dir = self.gen_dir
        # self.save_result(item_name,gen_dir, f0, ph_bd, tech_gt, tech_probs, tech_pred, tech_logits, threshold)
        self.saving_result_pool.add_job(self.save_result, args=[
            item_name, gen_dir, f0, ph, ph_bd, None, tech_probs, tech_pred, tech_logits, threshold])
        return {}

    @staticmethod
    def save_result(item_name, gen_dir, gt_f0=None, ph=None, ph_bd=None, tech_gt=None, tech_probs=None, tech_pred=None,
                    tech_logits=None, threshold=0.8):
        tech_names = ['mix', 'falsetto', 'breathe', 'bubble', 'strong', 'weak']
        for tech_idx, tech_name in enumerate(tech_names):
            print(item_name, tech_name)
            fig = f0_tech_txt_to_figure(gt_f0.numpy(), ph, ph_bd.numpy(), tech_pred[:, tech_idx].numpy(), None,
                                        tech_probs[:, tech_idx].numpy(), tech_name, fig_name=item_name,
                                        save_path=f'{gen_dir}/plot/{item_name}[{tech_name}].png')
            plt.close(fig)

    def test_end(self, outputs):
        pass


def bd_to_idxs(bd):
    # bd [T]
    idxs = []
    for idx in range(len(bd)):
        if bd[idx] == 1:
            idxs.append(idx)
    return idxs

def f0_tech_to_figure(f0_gt, ph_bd, tech_pred, tech_gt, tech_probs, tech_name, fig_name='', save_path=None):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(f0_gt, color='r', label='gt f0', alpha=0.7)
    ph_idxs = [0] + bd_to_idxs(ph_bd) + [len(ph_bd)]
    t_pred = np.zeros(f0_gt.shape[0])
    t_gt = np.zeros(f0_gt.shape[0])
    t_logits = np.zeros(f0_gt.shape[0])
    # print('-'*40)
    # print('f0_gt.shape[0]', f0_gt.shape[0])
    # print('ph_idxs', ph_idxs)
    for i in range(len(ph_idxs)-1):
        t_pred[ph_idxs[i]: ph_idxs[i + 1]] = tech_pred[i] * 200
        t_gt[ph_idxs[i]: ph_idxs[i + 1]] = tech_gt[i] * 200
        t_logits[ph_idxs[i]: ph_idxs[i + 1]] = tech_probs[i] * 200
        # print(tech_pred[i] * 200, tech_gt[i] * 200, tech_probs[i] * 200)
    plt.plot(t_gt, color='blue', label=f"gt {tech_name}", linestyle='--', alpha=0.7)
    plt.plot(t_pred, color='green', label=f"pred {tech_name}", linestyle='-.', alpha=0.7)
    plt.plot(t_logits, color='orange', label=f"logits {tech_name}", linestyle=':', alpha=0.7)
    plt.title(fig_name)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, format='png')
    return fig

def f0_tech_txt_to_figure(f0_gt, ph, ph_bd, tech_pred, tech_gt, tech_probs, tech_name, fig_name='', save_path=None):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(f0_gt, color='r', label='gt f0')
    ph_idxs = [0] + bd_to_idxs(ph_bd) + [len(ph_bd)]
    t_pred = np.zeros(f0_gt.shape[0])
    t_logits = np.zeros(f0_gt.shape[0])
    # print('-'*40)
    # print('f0_gt.shape[0]', f0_gt.shape[0])
    # print('ph_idxs', ph_idxs)
    for i in range(len(ph_idxs) - 1):
        shift = (i % 8) + 1
        if ph != "":
            plt.text(ph_idxs[i], shift * 4, ph[i])
            plt.vlines(ph_idxs[i], 0, 40, colors='b')
        t_pred[ph_idxs[i]: ph_idxs[i + 1]] = tech_pred[i] * 200
        t_logits[ph_idxs[i]: ph_idxs[i + 1]] = tech_probs[i] * 200
    plt.plot(t_pred, color='green', label=f"pred {tech_name}")
    plt.plot(t_logits, color='orange', label=f"logits {tech_name}")
    plt.title(fig_name)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, format='png')
    return fig

