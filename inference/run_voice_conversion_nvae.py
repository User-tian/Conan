#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

# Add inference directory to path
from inference.Conan import StreamingVoiceConversion

from utils.commons.hparams import hparams, set_hparams
from utils.audio.io import save_wav

class VoiceConversionRunner:
    """Run voice conversion on all prepared pairs"""
    
    def __init__(self, config_file='voice_conversion_config.json', hparams=None):
        self.config_file = config_file
        self.config = self.load_config()
        self.output_dir = "voice_conversion_output"
        self.setup_output_dir()
        
        # Initialize the voice conversion engine
        print("Initializing StreamingVoiceConversion engine...")
        self.engine = StreamingVoiceConversion(hparams)
        print("Engine initialized successfully!")
        _ = input()
        
    def load_config(self):
        """Load conversion pairs configuration"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
            
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            
        print(f"Loaded {config['total_pairs']} conversion pairs")
        return config
        
    def setup_output_dir(self):
        """Create output directory with timestamp"""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"test_output_nvae_conan"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
    def run_single_conversion(self, pair, pair_idx, spk_emb=None):
        """Run voice conversion for a single pair"""
        try:
            # Prepare input for Conan inference
            inp = {
                'ref_wav': pair['ref_wav'],
                'src_wav': pair['src_wav']
            }
            
            # Run inference
            wav_pred, mel_pred = self.engine.infer_once(inp, spk_emb=spk_emb)
            
            # Save output with specified naming
            output_name = f"{pair['src_corpus']}_{pair['src_utt_id']}.wav"
            output_path = os.path.join(self.output_dir, output_name)
            save_wav(wav_pred, output_path, hparams["audio_sample_rate"])
            
            return True, output_path
            
        except Exception as e:
            print(f"ERROR in pair {pair_idx}: {str(e)}")
            return False, str(e)
    
    def run_all_conversions(self, start_idx=0, end_idx=None, batch_size=50, embs=None):
        """Run voice conversion for all pairs with progress tracking"""
        pairs = self.config['conversion_pairs']
        total_pairs = len(pairs)
        
        if end_idx is None:
            end_idx = total_pairs

        assert len(embs) >= end_idx - start_idx
        print(f"Running conversions from {start_idx} to {end_idx} (total: {end_idx - start_idx})")
        
        # Statistics tracking
        successful = 0
        failed = 0
        start_time = time.time()
        errors = []
        
        # Progress file
        progress_file = os.path.join(self.output_dir, "conversion_progress.json")
        
        for i in range(start_idx, end_idx):
            pair = pairs[i]
            print(f"\nProcessing [{i+1}/{total_pairs}] {pair['output_name']}")
            print(f"  Src: {os.path.basename(pair['src_wav'])}")
            
            # Run conversion
            success, result = self.run_single_conversion(pair, i, spk_emb=embs[i])
            
            if success:
                successful += 1
                print(f"  ✓ Saved: {result}")
            else:
                failed += 1
                errors.append(f"Pair {i}: {result}")
                print(f"  ✗ Failed: {result}")
            
            # Save progress every batch_size files
            if (i - start_idx + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i - start_idx + 1)
                remaining = (end_idx - i - 1) * avg_time
                
                progress = {
                    'processed': i - start_idx + 1,
                    'total': end_idx - start_idx,
                    'successful': successful,
                    'failed': failed,
                    'elapsed_time': elapsed,
                    'estimated_remaining': remaining,
                    'current_batch_end': i,
                    'errors': errors
                }
                
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
                
                print(f"\n--- Progress Update ---")
                print(f"Processed: {i - start_idx + 1}/{end_idx - start_idx}")
                print(f"Success rate: {successful}/{i - start_idx + 1} ({100*successful/(i - start_idx + 1):.1f}%)")
                print(f"Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")
                print(f"Avg time per file: {avg_time:.1f}s")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n=== Final Summary ===")
        print(f"Total processed: {end_idx - start_idx}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {100*successful/(end_idx - start_idx):.1f}%")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per file: {total_time/(end_idx - start_idx):.1f}s")
        print(f"Output directory: {self.output_dir}")
        
        # Save final report
        final_report = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_processed': end_idx - start_idx,
            'successful': successful,
            'failed': failed,
            'success_rate': 100 * successful / (end_idx - start_idx),
            'total_time_minutes': total_time / 60,
            'avg_time_per_file_seconds': total_time / (end_idx - start_idx),
            'output_directory': self.output_dir,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, "final_report.json"), 'w') as f:
            json.dump(final_report, f, indent=2)
        
        if failed > 0:
            print(f"\nErrors encountered:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

def main():
    """Main function"""
    # Set hparams first before creating the runner
    set_hparams(config="egs/extract_spk_emb.yaml", exp_name="render_nvae")

    spk_embs = np.load("/storageSSD/huiran/src/NVAE-DarkStream/output/nvae_conan/emb_900.npy")
    
    # Create runner and execute with default parameters
    runner = VoiceConversionRunner('voice_conversion_config.json', hparams=hparams)
    runner.run_all_conversions(start_idx=0, end_idx=None, batch_size=50, embs=spk_embs)

if __name__ == "__main__":
    main() 
