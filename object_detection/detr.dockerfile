FROM determinedai/environments:cuda-10.0-pytorch-1.4-tf-1.15-gpu-067db2b
RUN apt-get update
RUN apt-get -y install unzip curl && rm -rf /var/lib/apt/lists/*

# Install docker client    
ENV DOCKER_CHANNEL stable
ENV DOCKER_VERSION 20.10.3
ENV DOCKER_API_VERSION 1.41
RUN curl -fsSL "https://download.docker.com/linux/static/${DOCKER_CHANNEL}/x86_64/docker-${DOCKER_VERSION}.tgz" \
  | tar -xzC /usr/local/bin --strip=1 docker/docker

# Download data and install dependencies
RUN wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
RUN unzip -o annotations_trainval2017.zip 
RUN mv annotations/instances_train2017.json /tmp
RUN mv annotations/instances_val2017.json /tmp

RUN pip install tqdm attrdict pycocotools cython scipy
RUN conda install -y jupyter

RUN git clone https://github.com/facebookresearch/detr.git
RUN cd detr && git reset --hard 4e1a9281bc5621dcd65f3438631de25e255c4269 && cd ..

RUN git clone https://github.com/fundamentalvision/Deformable-DETR ddetr
RUN cd ddetr && git reset --hard 11169a60c33333af00a4849f1808023eba96a931 && cd ..

RUN cd ddetr/models/ops  && ./make.sh && cd ../../..
