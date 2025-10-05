# extract f0 rmvpe
python trials/extract_f0_rmvpe.py \
    --input_dir /work/hdd/bcza/usertian/vctk-controlvc16k/wav16_silence_trimmed_padded\
    --output_dir /work/hdd/bcza/usertian/vctk-controlvc16k/wav16_silence_trimmed_padded_f0


python data_gen/tts/runs/binarize.py --config egs/emformer.yaml

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/stage1.yaml  --exp_name trial

CUDA_VISIBLE_DEVICES=1 python inference/techsinger.py --config egs/stage1.yaml  --exp_name checkpoints/stage1_minimal


python trials/update_hubert_in_metadata.py \
    --metadata data/processed/vc/metadata_emformer_wholesen.json \
    --mapping /storageNVME/baotong/datasets/vctk-controlvc16k/hubert_distillation_w_libritts_vctk_6layer_rc2.txt \
    --output data/processed/vc/metadata_6layer.json

# train emformer
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config egs/emformer.yaml  --exp_name emformer_test2 --reset
# run main model Conan
CUDA_VISIBLE_DEVICES=2 python tasks/run.py --config egs/stage1_emformer.yaml  --exp_name conan_test --reset

# run hifigan
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config egs/hifinsf_16k320_shuffle.yaml  --exp_name hifigan_test --reset



CUDA_VISIBLE_DEVICES=1 python inference/techsinger.py --config egs/stage1.yaml --exp_name stage1_clean


# whole system Conan inference
CUDA_VISIBLE_DEVICES=1 python inference/Conan.py --config egs/stage1_emformer.yaml --exp_name conan_mainmodeltest

CUDA_VISIBLE_DEVICES=2 python inference/Conan_previousemformer.py --config egs/stage1_previousemformer.yaml --exp_name stage1_clean


CUDA_VISIBLE_DEVICES=0 python inference/gradio_realtime_demo.py --config egs/stage1_emformer.yaml --exp_name conan_mainmodeltest # not successful

CUDA_VISIBLE_DEVICES=0 python inference/run_voice_conversion.py --config egs/stage1_previousemformer.yaml --exp_name stage1_clean

CUDA_VISIBLE_DEVICES=0 python test_vc_metrics.py --test_output test_output_previousemformer --test_dataset test_dataset --output_dir results_previousemformer

