# extract f0 rmvpe
python trials/extract_f0_rmvpe.py \
    --input_dir /work/hdd/bcza/usertian/vctk-controlvc16k/wav16_silence_trimmed_padded\
    --output_dir /work/hdd/bcza/usertian/vctk-controlvc16k/wav16_silence_trimmed_padded_f0


python data_gen/tts/runs/binarize.py --config egs/stage1.yaml

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/stage1.yaml  --exp_name trial

CUDA_VISIBLE_DEVICES=1 python inference/techsinger.py --config egs/stage1.yaml  --exp_name checkpoints/stage1_minimal


python trials/update_hubert_in_metadata.py \
    --metadata data/processed/vc/metadata_emformer_wholesen.json \
    --mapping /storageNVME/baotong/datasets/vctk-controlvc16k/hubert_distillation_w_libritts_vctk_6layer_rc2.txt \
    --output data/processed/vc/metadata_6layer.json

CUDA_VISIBLE_DEVICES=2 python tasks/run.py --config egs/stage1.yaml  --exp_name stage1_minimal

CUDA_VISIBLE_DEVICES=1 python inference/techsinger.py --config egs/stage1.yaml --exp_name stage1_clean


# whole system Conan
CUDA_VISIBLE_DEVICES=1 python inference/Conan.py --config egs/stage1_emformer.yaml --exp_name stage1_clean