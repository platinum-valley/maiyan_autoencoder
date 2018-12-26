export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --gpu \
    --task style_transfer\
    --generate_mode \
    --model_type VAEGAN\
    --generator_model weights_generator3.pth\
    --discriminator_model weight_discriminator3.pth\
    --classifier_model weight_classifier.pth\
    --train_data train_data.csv\
    --valid_data valid_data.csv\
    --style_image ./nogi_face/shiraishi_1.jpg\
    --output_dir output_img_VAEGAN\
    --lr 0.0001\
    --batch_size 32\
    --epochs 200
