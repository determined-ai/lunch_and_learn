apt-get update
apt-get install unzip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip 
mv annotations/instances_train2017.json /tmp
mv annotations/instances_val2017.json /tmp

git clone https://github.com/fundamentalvision/Deformable-DETR ddetr
cd ddetr && git reset --hard 11169a60c33333af00a4849f1808023eba96a931 

pip install tqdm attrdict pycocotools cython scipy

cd models/ops 
#sed 's/const int CUDA_NUM_THREADS = 1024;/const int CUDA_NUM_THREADS = 512;/' src/cuda/ms_deform_im2col_cuda.cuh
sh ./make.sh
#python test.py
cd ../../..
