FROM projectmonai/monai

RUN pip install nilearn

COPY ./ /opt/3DUnetCNN
ENV PYTHONPATH=/opt/3DUnetCNN:$PYTHONPATH
ENV PATH=/opt/3DUnetCNN/unet3d/scripts:$PATH
RUN chmod +x /opt/3DUnetCNN/unet3d/scripts/*.py

