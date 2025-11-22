python train.py --data_path Dataset/ --split 'place'
python train.py --data_path D:\\Cafe_Dataset\\Cafe_Dataset\\Dataset --split place --batch 4 --device "0"
python train.py --data_path /share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/ --split place
python train.py --data_path /share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/ --split place --drop_rate 0.15 --gar_nheads 8 --device "0, 1"   
python train.py --data_path /share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/ --split place --drop_rate 0.2 --gar_nheads 8 --device "2, 3" --num_frame 8 --batch 12
